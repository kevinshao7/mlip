"""Validate a read-only transfer snapshot; invoked through campaign.py."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
from ase.io import read, write

import orca_output_validation as validator


def export_package(frames, rows, ready, campaign):
    """A portable, Git-transferable dataset; no machine-specific source paths."""
    package = campaign.BASE_OUT / "acquisition_p10/hpc_dataset"
    package.mkdir(parents=True, exist_ok=True)
    seed = 20261029
    indices = list(range(len(frames)))
    random.Random(seed).shuffle(indices)
    count = max(1, min(len(frames)-1, round(len(frames)*0.2)))
    valid = sorted(indices[:count]); train = sorted(indices[count:])
    for name, members in (("processed.extxyz", frames),
                          ("train.extxyz", [frames[i] for i in train]),
                          ("validation.extxyz", [frames[i] for i in valid])):
        write(package / name, members, format="extxyz")
    manifest = {"ready_for_training": ready, "expected_outputs": 1000,
                "missing_outputs": sum(r["issues"] == "missing_output" for r in rows),
                "rejected_present_outputs": sum(r["status"] == "rejected" and r["issues"] != "missing_output" for r in rows),
                "accepted": len(frames), "validation_fraction": 0.2,
                "seed": seed, "train_indices": train, "validation_indices": valid,
                "train_count": len(train), "validation_count": len(valid),
                "files": {name: campaign.sha(package / name) for name in
                          ("processed.extxyz", "train.extxyz", "validation.extxyz")},
                "caveat": "Non-equilibrium, sub-target-pressure P10 compression transient",
                "foundation": "polar-1-l", "units": "total energy eV, forces eV/Angstrom; PBC false",
                "source_results": [{k: r[k] for k in ("stem", "status", "issues")} for r in rows]}
    campaign.write_json(package / "dataset.json", manifest)
    print(f"Portable dataset: {package} (ready_for_training={ready})")


def prepare(args):
    import campaign as campaign
    campaign.RUN_ID = "acquisition_p10"
    campaign.OUT = campaign.BASE_OUT / campaign.RUN_ID

    package = campaign.ROOT / "dft_launches/p10_2gpu_6h/P10p0000GPa/acquisition/inputs"
    source = args.source.resolve()
    inputs = sorted(package.glob("cluster_*.inp"))
    if len(inputs) != 1000:
        raise ValueError(f"Expected the approved 1000-input P10 package, found {len(inputs)}")
    destination = campaign.cycle(10.0) / "dft/acquisition"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = destination / "snapshots" / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    rows, accepted, hashes = [], [], {}
    for path in inputs:
        charge, spin, symbols, positions = validator.parse_input(path)
        output = source / f"{path.stem}.out"
        before = (output.stat().st_size, output.stat().st_mtime_ns) if output.exists() else None
        digest = campaign.sha(output) if before else None
        row, atoms = validator.process(validator.Job(
            int(path.stem.split("_")[-1]), path.stem, path, output,
            charge, spin, len(symbols)))
        after = (output.stat().st_size, output.stat().st_mtime_ns) if output.exists() else None
        if before != after:
            row.update(status="rejected", issues=str(row["issues"]) + ";changed_during_read")
            atoms = None
        # Bind the external stdout to the exact approved input geometry/method.
        if atoms is not None:
            text = output.read_text(errors="replace")
            echoed = "\n".join(line.split(">", 1)[1].strip() for line in text.splitlines()
                               if validator.re.match(r"^\s*\|\s*\d+>", line))
            temporary = snapshot / "echoed.inp"
            temporary.write_text(echoed, encoding="utf-8")
            try:
                eq, es, sy, pos = validator.parse_input(temporary)
                if (eq != charge or es != spin or sy != symbols
                        or not np.allclose(pos, positions, rtol=0, atol=1e-8)
                        or path.read_text().splitlines()[0] not in echoed):
                    raise ValueError("output geometry or method differs from approved input")
                if not (np.isfinite(atoms.info["REF_energy"])
                        and np.isfinite(atoms.arrays["REF_forces"]).all()):
                    raise ValueError("nonfinite energy or forces")
            except ValueError as exc:
                row.update(status="rejected", issues=str(exc))
                atoms = None
            finally:
                temporary.unlink()
        final_stat = (output.stat().st_size, output.stat().st_mtime_ns) if output.exists() else None
        if final_stat != before:
            row.update(status="rejected", issues=str(row["issues"]) + ";changed_during_read")
            atoms = None
        rows.append(row)
        if digest:
            hashes[output.name] = digest
        if atoms is not None:
            atoms.info.update(multiplicity=spin, acquisition_id=path.stem,
                              ensemble_caveat="non_equilibrium_sub_target_pressure_P10",
                              source_sha256=digest, input_sha256=campaign.sha(path))
            accepted.append(atoms)
    with (snapshot / "collection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source": str(source), "expected": len(inputs), "accepted": len(accepted),
               "rejected": len(inputs)-len(accepted), "output_sha256": hashes,
               "issues": dict(Counter(str(r["issues"]) for r in rows if r["status"] != "accepted")),
               "units": {"REF_energy": "eV total", "REF_forces": "eV/Angstrom"},
               "validation_fraction": 0.2, "raw_files_preserved": True}
    campaign.write_json(snapshot / "summary.json", summary)
    if accepted:
        write(snapshot / "processed.extxyz", accepted, format="extxyz")
        # Round-trip validation prevents inconsistent extxyz entering training.
        for atoms in read(snapshot / "processed.extxyz", ":"):
            expected = sum(validator.FORMAL_CHARGES[s] for s in atoms.get_chemical_symbols())
            if atoms.info["charge"] != expected or atoms.info["multiplicity"] != 1:
                raise ValueError("Condensed extxyz charge/spin validation failed")
    campaign.event("transfer_snapshot_validated", 10.0, inputs, [snapshot / "summary.json"],
                   accepted=len(accepted), expected=len(inputs))
    print(json.dumps({k: v for k, v in summary.items() if k != "output_sha256"}, indent=2))
    print(f"Audit snapshot: {snapshot}")
    if len(accepted) >= 30:
        export_package(accepted, rows, len(hashes) == len(inputs) or args.allow_partial, campaign)
    if len(accepted) < 30:
        raise ValueError("Fewer than 30 validated labels; audit saved, training preparation refused")
    if len(hashes) != len(inputs) and not args.allow_partial:
        print("Transfer incomplete. Audit saved; rerun after copying, or explicitly use --allow-partial.")
        return
    campaign.command_init(args)
    cfg = campaign.config()
    cfg["finetune"]["validation_fraction"] = 0.2
    campaign.write_json(campaign.OUT / "campaign_config.json", cfg)
    # Publish only validated labels; prior generated collections stay auditable.
    import shutil
    published = destination / "processed.extxyz"
    if published.exists():
        shutil.copy2(published, snapshot / "previous_processed.extxyz")
    shutil.copy2(snapshot / "processed.extxyz", published)
    args.pressure = 10.0
    args.run = False
    args.epochs = None
    args.lr = 1e-4
    args.parent_model = "polar-1-l"
    args.changed_parameter = "acquisition_dataset"
    args.change_note = "Approved P10 DFT acquisition; polar-1-l, 80/20 split, unchanged baseline optimization."
    campaign.command_finetune(args)
