#!/usr/bin/env python3
"""Compare polar-1-s foundation, MACE-OFF medium, and best fine-tuned campaign model on DFT data.

Force plot: |F_DFT| per atom (x) vs |ΔF| = |F_pred - F_DFT| per atom (y), element-coloured.
Energy plot: DFT reference energy (x) vs model predicted energy (y).
  - Polar models share the same E0 reference as the DFT training data.
  - MACE-OFF energies are shifted to the same E0 reference using target_dft_e0s.json.

Usage (from X_semiactivelearning/):
    /home/kevinsh/env/bin/python compare_foundation.py --dataset all
    /home/kevinsh/env/bin/python compare_foundation.py --dataset all --skip-evaluation
    /home/kevinsh/env/bin/python compare_foundation.py --no-maceoff --dataset all
    /home/kevinsh/env/bin/python compare_foundation.py --finetuned-model /path/to/model.model

Outputs land in outputsfull/X_semiactivelearning/foundation_comparison/<dataset>/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_MPL_CACHE = Path(__file__).resolve().parent / "runs" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

CODES = Path(__file__).resolve().parent
MLIP  = CODES.parents[1]
OUT   = MLIP / "outputsfull" / "X_semiactivelearning"
D2    = MLIP / "codes" / "D2_naive"

DATASETS = {
    "p31_r2":  OUT / "cycles" / "P31p6228GPa" / "dft" / "acquisition_r2" / "processed.extxyz",
    "p31_all": None,
    "all":     OUT / "cycles" / "P31p6228GPa" / "training" / "cumulative_all.extxyz",
    "val":     OUT / "cycles" / "P31p6228GPa" / "training" / "random_validation.extxyz",
}

# Colorblind-safe Wong palette per element
ELEMENT_COLORS = {"H": "#56B4E9", "N": "#D55E00", "O": "#009E73"}
ELEMENT_OTHER_COLOR = "#CC79A7"
ELEMENT_ORDER = ["H", "N", "O"]

# Distinct colours for each model panel title / stat box
MODEL_TITLE_COLORS = ["#0072B2", "#E69F00", "#009E73"]   # blue, amber, green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_finetuned_model() -> Path:
    lineage = json.loads((OUT / "model_lineage.json").read_text(encoding="utf-8"))
    active  = str(lineage.get("active_branch", ""))
    history = [h for h in lineage.get("history", []) if str(h.get("branch_id", "")) == active]
    if not history:
        raise RuntimeError(f"No fine-tuned models on active branch {active!r}")
    best  = min(history, key=lambda h: h.get("validation_force_rmse_mev_per_a", float("inf")))
    model = Path(best["model"])
    if not model.is_file():
        raise FileNotFoundError(f"Best model file not found: {model}")
    print(f"[compare] Fine-tuned: {best['finetune_id']}  "
          f"lr={best.get('learning_rate')}  val_F={best.get('validation_force_rmse_mev_per_a')} meV/Å")
    return model


def build_p31_all(tmp: Path) -> Path:
    import ase.io
    out_path = tmp / "p31_all.extxyz"
    if out_path.exists():
        return out_path
    frames = []
    for kind in ("acquisition", "acquisition_r2"):
        p = OUT / "cycles" / "P31p6228GPa" / "dft" / kind / "processed.extxyz"
        if p.is_file():
            frames.extend(ase.io.read(str(p), index=":"))
    if not frames:
        raise FileNotFoundError("No P31 acquisition data found")
    tmp.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(out_path), frames, format="extxyz")
    print(f"[compare] Built p31_all: {len(frames)} frames → {out_path}")
    return out_path


def resolve_foundation(model_str: str) -> Path:
    local = Path(model_str).expanduser()
    if local.is_file():
        return local.resolve()
    sys.path.insert(0, str((MLIP / "mace").resolve()))
    from mace.calculators.foundations_models import download_mace_polar_checkpoint
    return Path(download_mace_polar_checkpoint(model_str)).resolve()


def run_evaluate(configs: Path, model: Path | str, output: Path, device: str) -> None:
    cmd = [sys.executable, str(D2 / "evaluate.py"),
           "--configs", str(configs), "--model", str(model),
           "--output", str(output), "--device", device]
    print("[compare] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


def run_evaluate_maceoff(configs: Path, output: Path, device: str) -> None:
    """Evaluate MACE-OFF medium; write MACE_energy + MACE_forces to extxyz."""
    import numpy as np
    from ase.io import read as ase_read, write as ase_write
    from mace.calculators import mace_off

    print("[compare] Evaluating MACE-OFF medium ...")
    calc = mace_off(model="medium", device=device)
    frames_in  = ase_read(str(configs), index=":")
    frames_out = []
    for atoms in frames_in:
        a = atoms.copy()
        a.calc = calc
        energy = float(a.get_potential_energy())
        forces = a.get_forces().copy()
        a.calc = None   # detach — prevents internal MACE arrays (wrong shape) being written
        a.info["MACE_energy"]   = energy
        a.arrays["MACE_forces"] = forces
        frames_out.append(a)
    output.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output), frames_out, format="extxyz")
    print(f"[compare] MACE-OFF predictions → {output}")


def get_maceoff_e0s(device: str) -> dict[int, float]:
    """Return {atomic_number: E0_eV} for MACE-OFF medium."""
    from mace.calculators import mace_off
    calc = mace_off(model="medium", device=device)
    model = calc.models[0]
    Z  = model.atomic_numbers.cpu().tolist()
    E0 = model.atomic_energies_fn.atomic_energies.detach().cpu().flatten().tolist()
    return dict(zip(Z, E0))


def get_dft_e0s() -> dict[int, float]:
    """Return {atomic_number: E0_eV} from target_dft_e0s.json (used for polar-1-s training)."""
    p = D2 / "data" / "target_dft_e0s.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in raw.items()}


def maceoff_energy_correction(frames_dft, maceoff_e0s: dict, dft_e0s: dict) -> "np.ndarray":
    """Per-structure additive correction to align MACE-OFF to the DFT E0 reference.

    REF_energy = E_DFT - sum(dft_E0[Z])
    MACE-OFF reports E_raw (total energy in its own frame, including its own E0s).
    Corrected MACE-OFF = E_raw - sum(maceoff_E0[Z]) + sum(dft_E0[Z])
                       = E_raw + correction_per_structure
    correction = sum((dft_E0[Z] - maceoff_E0[Z]) for Z in structure)
    """
    import numpy as np
    from ase.data import atomic_numbers as ASE_Z
    corrections = []
    for atoms in frames_dft:
        c = sum(
            dft_e0s.get(ASE_Z[sym], 0.0) - maceoff_e0s.get(ASE_Z[sym], 0.0)
            for sym in atoms.get_chemical_symbols()
        )
        corrections.append(c)
    return np.array(corrections)   # eV, add to MACE-OFF raw energy to get comparable energy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_by_element(ax, x, y, elements_arr, unique_elements, worst_set):
    import numpy as np
    for elem in unique_elements:
        mask = elements_arr == elem
        if not mask.any():
            continue
        color = ELEMENT_COLORS.get(elem, ELEMENT_OTHER_COLOR)
        idx   = np.where(mask)[0]
        reg   = np.array([i for i in idx if i not in worst_set])
        out   = np.array([i for i in idx if i in worst_set])
        if len(reg):
            ax.scatter(x[reg], y[reg], s=14, alpha=0.65, color=color,
                       edgecolors="none", label=elem, zorder=2)
        if len(out):
            ax.scatter(x[out], y[out], s=100, alpha=0.95, color=color,
                       marker="x", linewidths=1.8, label=f"{elem} (outlier)", zorder=5)


def parity_plot_forces(
    dft_mag: "np.ndarray",
    models: "list[tuple[np.ndarray, str]]",   # [(err_mag_array, label), ...]
    elements: "list[str]",
    title_suffix: str,
    output: Path,
    outliers: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(models)
    elements_arr    = np.array(elements)
    unique_elements = ELEMENT_ORDER + sorted(set(elements) - set(ELEMENT_ORDER))

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6.5), constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    for ax, (err_mag, label), tcol in zip(axes, models, MODEL_TITLE_COLORS):
        mean_err = float(np.mean(err_mag))
        rmse_err = float(np.sqrt(np.mean(err_mag ** 2)))
        x_hi = float(np.percentile(dft_mag, 99)) * 1.05
        y_hi = float(np.percentile(err_mag, 99)) * 1.15

        worst = set(np.argsort(err_mag)[-min(outliers, len(err_mag)):].tolist())
        _scatter_by_element(ax, dft_mag, err_mag, elements_arr, unique_elements, worst)

        ax.axhline(mean_err, color="#444444", lw=1.0, ls="--", zorder=0,
                   label=f"mean {mean_err:.1f}")
        ax.set(xlim=(0, x_hi), ylim=(0, y_hi),
               xlabel="|F_DFT| per atom (meV/Å)",
               ylabel="|ΔF| = |F_pred − F_DFT| (meV/Å)",
               title=f"{label}\n{title_suffix}")
        ax.title.set_color(tcol)
        ax.text(0.97, 0.97,
                f"N={len(dft_mag)} atoms\nmean |ΔF|={mean_err:.2f}\nRMSE={rmse_err:.2f} meV/Å",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9})

        handles, labels_leg = ax.get_legend_handles_labels()
        seen, dh, dl = set(), [], []
        for h, l in zip(handles, labels_leg):
            k = l.replace(" (outlier)", "")
            if k not in seen:
                seen.add(k); dh.append(h); dl.append(l)
        ax.legend(dh, dl, title="Element  (× = outlier)", fontsize=8, loc="upper left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parity_plot_energy(
    dft_E: "np.ndarray",
    models: "list[tuple[np.ndarray, str]]",   # [(pred_energy_array, label), ...]
    title_suffix: str,
    output: Path,
    outliers: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 6), constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    for ax, (pred, label), tcol in zip(axes, models, MODEL_TITLE_COLORS):
        err  = pred - dft_E
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae  = float(np.mean(np.abs(err)))
        lo   = min(dft_E.min(), pred.min())
        hi   = max(dft_E.max(), pred.max())
        pad  = max((hi - lo) * 0.04, 1e-8)
        worst = np.argsort(np.abs(err))[-min(outliers, len(err)):][::-1]

        ax.scatter(dft_E, pred, s=20, alpha=0.7, color=tcol, edgecolors="none")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", lw=1.2, zorder=0)
        ax.scatter(dft_E[worst], pred[worst], s=60, color="#000000",
                   marker="x", linewidths=1.5, zorder=3, label="outlier")
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
               xlabel="DFT energy (meV)", ylabel="Model energy (meV)",
               title=f"{label}\n{title_suffix}")
        ax.title.set_color(tcol)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.03, 0.97, f"N={len(dft_E)}\nRMSE={rmse:.4g} meV\nMAE={mae:.4g} meV",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9})

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), default="p31_r2")
    p.add_argument("--configs", type=Path, default=None)
    p.add_argument("--finetuned-model", type=Path, default=None)
    p.add_argument("--foundation-model", default="polar-1-s")
    p.add_argument("--no-maceoff", action="store_true", help="Skip MACE-OFF evaluation")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--outliers", type=int, default=12)
    p.add_argument("--skip-evaluation", action="store_true",
                   help="Reuse existing prediction xyz files")
    return p.parse_args()


def main() -> None:
    import numpy as np
    from ase.io import read as ase_read

    args = parse_args()
    finetuned   = args.finetuned_model or best_finetuned_model()
    use_maceoff = not args.no_maceoff

    tmp = OUT / "foundation_comparison" / ".tmp"
    if args.configs is not None:
        configs      = args.configs.resolve()
        dataset_name = configs.stem
    elif args.dataset == "p31_all":
        configs      = build_p31_all(tmp)
        dataset_name = "p31_all"
    else:
        configs = DATASETS[args.dataset]
        if configs is None:
            raise RuntimeError(f"Dataset {args.dataset!r} has no static path")
        if not configs.is_file():
            raise FileNotFoundError(f"Dataset not found: {configs}")
        dataset_name = args.dataset

    output_dir = OUT / "foundation_comparison" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    foundation_pred = output_dir / "foundation_predictions.xyz"
    finetuned_pred  = output_dir / "finetuned_predictions.xyz"
    maceoff_pred    = output_dir / "maceoff_predictions.xyz"

    if not args.skip_evaluation:
        foundation_path = resolve_foundation(args.foundation_model)
        run_evaluate(configs, foundation_path, foundation_pred, args.device)
        run_evaluate(configs, finetuned,       finetuned_pred,  args.device)
        if use_maceoff:
            run_evaluate_maceoff(configs, maceoff_pred, args.device)

    for p in ([foundation_pred, finetuned_pred] +
              ([maceoff_pred] if use_maceoff else [])):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing prediction file — run without --skip-evaluation: {p}")

    dft_frames = ase_read(str(configs),        index=":")
    fnd_frames = ase_read(str(foundation_pred), index=":")
    tnd_frames = ase_read(str(finetuned_pred),  index=":")
    off_frames = ase_read(str(maceoff_pred),    index=":") if use_maceoff else None

    n = len(dft_frames)

    # --- energies (meV) ---
    dft_E = 1000.0 * np.array([f.info["REF_energy"]  for f in dft_frames])
    fnd_E = 1000.0 * np.array([f.info["MACE_energy"] for f in fnd_frames])
    tnd_E = 1000.0 * np.array([f.info["MACE_energy"] for f in tnd_frames])

    energy_models = [
        (fnd_E, "polar-1-s foundation"),
        (tnd_E, "Fine-tuned polar (best)"),
    ]

    if use_maceoff:
        off_E_raw = 1000.0 * np.array([f.info["MACE_energy"] for f in off_frames])
        # Align MACE-OFF to the DFT E0 reference used for polar-1-s training
        maceoff_e0s = get_maceoff_e0s(args.device)
        dft_e0s     = get_dft_e0s()
        correction  = 1000.0 * maceoff_energy_correction(dft_frames, maceoff_e0s, dft_e0s)
        off_E = off_E_raw + correction
        energy_models.append((off_E, "MACE-OFF medium (E0-corrected)"))

    # --- forces (meV/Å, per-atom vectors) ---
    dft_F_list, fnd_F_list, tnd_F_list, elem_list = [], [], [], []
    off_F_list = [] if use_maceoff else None
    for i, (dft, fnd, tnd) in enumerate(zip(dft_frames, fnd_frames, tnd_frames)):
        dft_F_list.append(dft.arrays["REF_forces"])
        fnd_F_list.append(fnd.arrays["MACE_forces"])
        tnd_F_list.append(tnd.arrays["MACE_forces"])
        elem_list.extend(dft.get_chemical_symbols())
        if use_maceoff:
            off_F_list.append(off_frames[i].arrays["MACE_forces"])

    dft_Fvec = 1000.0 * np.concatenate(dft_F_list, axis=0)
    fnd_Fvec = 1000.0 * np.concatenate(fnd_F_list, axis=0)
    tnd_Fvec = 1000.0 * np.concatenate(tnd_F_list, axis=0)

    dft_Fmag = np.linalg.norm(dft_Fvec, axis=1)
    fnd_dF   = np.linalg.norm(fnd_Fvec - dft_Fvec, axis=1)
    tnd_dF   = np.linalg.norm(tnd_Fvec - dft_Fvec, axis=1)

    force_models = [
        (fnd_dF, "polar-1-s foundation"),
        (tnd_dF, "Fine-tuned polar (best)"),
    ]

    if use_maceoff:
        off_Fvec = 1000.0 * np.concatenate(off_F_list, axis=0)
        off_dF   = np.linalg.norm(off_Fvec - dft_Fvec, axis=1)
        force_models.append((off_dF, "MACE-OFF medium"))

    n_atoms      = len(dft_Fmag)
    title_suffix = f"{dataset_name}  ({n} configs, {n_atoms} atoms)"

    parity_plot_energy(dft_E, energy_models, title_suffix,
                       output_dir / "energy_parity.png", args.outliers)
    parity_plot_forces(dft_Fmag, force_models, elem_list, title_suffix,
                       output_dir / "force_parity.png", args.outliers)

    # --- summary ---
    summary  = {}
    elem_arr = np.array(elem_list)
    for name, pred_E, dF in (
        [("foundation", fnd_E, fnd_dF), ("finetuned", tnd_E, tnd_dF)] +
        ([("maceoff", off_E, off_dF)] if use_maceoff else [])
    ):
        summary[name] = {
            "energy_rmse_meV":          float(np.sqrt(np.mean((pred_E - dft_E) ** 2))),
            "energy_mae_meV":           float(np.mean(np.abs(pred_E - dft_E))),
            "force_err_mean_meV_per_A": float(np.mean(dF)),
            "force_err_rmse_meV_per_A": float(np.sqrt(np.mean(dF ** 2))),
        }
        for elem in sorted(set(elem_list)):
            summary[name][f"force_err_mean_{elem}_meV_per_A"] = float(np.mean(dF[elem_arr == elem]))

    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    models_ordered = ["foundation", "finetuned"] + (["maceoff"] if use_maceoff else [])
    print(f"\n{'':=<72}")
    print(f"  {'metric':<36}" + "".join(f"{m:>12}" for m in models_ordered))
    print(f"{'':=<72}")
    for metric in ("energy_rmse_meV", "force_err_mean_meV_per_A", "force_err_rmse_meV_per_A"):
        row = f"  {metric:<36}" + "".join(f"{summary[m][metric]:12.2f}" for m in models_ordered)
        print(row)
    for elem in ELEMENT_ORDER:
        key = f"force_err_mean_{elem}_meV_per_A"
        if key in summary["foundation"]:
            row = f"  {'mean |ΔF| ' + elem:<36}" + "".join(
                f"{summary[m][key]:12.2f}" for m in models_ordered)
            print(row)
    print(f"{'':=<72}")
    print(f"\nPlots   → {output_dir}/energy_parity.png")
    print(f"           {output_dir}/force_parity.png")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
