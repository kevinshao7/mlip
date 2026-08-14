#!/usr/bin/env python3
"""Generate BlueHive ORCA input and Slurm files from base templates.

Submit generated jobs with:
    for f in expand/*.slurm; do sbatch "$f"; done
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_INP = SCRIPT_DIR / "base.inp"
BASE_SLURM = SCRIPT_DIR / "base.slurm"
OUT_DIR = SCRIPT_DIR / "expand"
DEFAULT_CLUSTER_XYZ = (
    SCRIPT_DIR.parent
    / "7_26_H2pathvalidation"
    / "r09_hot_w_h2formation_training_clusters.xyz"
)
STEM_PREFIX = "r09_hot_w_isolatedH_bluehive"
MULTIPLICITY = 1

FORMAL_CHARGES = {
    "H": 1,
    "N": -3,
    "O": -2,
    "S": -2,
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_frames(spec: str) -> tuple[int, int]:
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) != 2:
        fail(f"--frames must be start,stop, got {spec!r}")
    start, stop = (int(fields[0]), int(fields[1]))
    if start < 0 or stop <= start:
        fail(f"--frames must satisfy 0 <= start < stop, got {spec!r}")
    return start, stop


def stem_for_frame(frame_index: int) -> str:
    return f"{STEM_PREFIX}_{frame_index:03d}"


def formal_charge(symbols: list[str]) -> int:
    charge = 0
    for symbol in symbols:
        try:
            charge += FORMAL_CHARGES[symbol]
        except KeyError as exc:
            raise ValueError(f"No formal charge configured for {symbol!r}") from exc
    return charge


def coordinate_block(atoms) -> str:
    rows = []
    for symbol, x, y, z in atoms:
        rows.append(f"{symbol:<2}  {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(rows)


def read_xyz_frames(path: Path) -> list[list[tuple[str, float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[list[tuple[str, float, float, float]]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            n_atoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {cursor + 1} in {path}") from exc
        cursor += 2
        if cursor + n_atoms > len(lines):
            raise ValueError(f"Truncated XYZ frame ending after line {cursor + 1} in {path}")

        frame: list[tuple[str, float, float, float]] = []
        for line in lines[cursor : cursor + n_atoms]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed XYZ atom line: {line!r}")
            frame.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        frames.append(frame)
        cursor += n_atoms
    return frames


def render_template(template: str, values: dict[str, str]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    unresolved = sorted(set(re.findall(r"{{[A-Z_]+}}", text)))
    if unresolved:
        fail(f"Unresolved template placeholders: {', '.join(unresolved)}")
    return text


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def remove_stale_generated() -> None:
    for pattern in (f"{STEM_PREFIX}_*.inp", f"{STEM_PREFIX}_*.slurm"):
        for path in OUT_DIR.glob(pattern):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="0,180", help="Half-open frame range, e.g. 0,180")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_XYZ)
    parser.add_argument("--clean", action="store_true", help="Remove stale generated files in expand/ first.")
    args = parser.parse_args()

    if not BASE_INP.is_file():
        fail(f"Missing input template: {BASE_INP}")
    if not BASE_SLURM.is_file():
        fail(f"Missing Slurm template: {BASE_SLURM}")
    if not args.clusters.is_file():
        fail(f"Missing cluster XYZ: {args.clusters}")

    start, stop = parse_frames(args.frames)
    frames = read_xyz_frames(args.clusters)
    if len(frames) < stop:
        fail(f"Requested frames {start},{stop}, but {args.clusters} has only {len(frames)} frames")

    OUT_DIR.mkdir(exist_ok=True)
    if args.clean:
        remove_stale_generated()

    inp_template = BASE_INP.read_text(encoding="utf-8")
    slurm_template = BASE_SLURM.read_text(encoding="utf-8")

    manifest_lines = ["frame,stem,input,slurm,charge,multiplicity,n_atoms"]
    for frame_index, atoms in zip(range(start, stop), frames[start:stop]):
        stem = stem_for_frame(frame_index)
        symbols = [symbol for symbol, _x, _y, _z in atoms]
        charge = formal_charge(symbols)
        values = {
            "STEM": stem,
            "CHARGE": str(charge),
            "MULTIPLICITY": str(MULTIPLICITY),
            "COORDINATES": coordinate_block(atoms),
        }

        inp_path = OUT_DIR / f"{stem}.inp"
        slurm_path = OUT_DIR / f"{stem}.slurm"
        write_text_lf(inp_path, render_template(inp_template, values))
        write_text_lf(slurm_path, render_template(slurm_template, values))
        manifest_lines.append(
            f"{frame_index},{stem},expand/{inp_path.name},expand/{slurm_path.name},"
            f"{charge},{MULTIPLICITY},{len(atoms)}"
        )
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")
        print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)}")

    write_text_lf(OUT_DIR / "manifest.csv", "\n".join(manifest_lines) + "\n")
    print(f"Generated {stop - start} input files and {stop - start} Slurm files in {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
