#!/usr/bin/env python3
"""Generate isolated-atom ORCA inputs with the C_DFTproduction settings."""

from __future__ import annotations

import shutil
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent / "C_DFTproduction"
PRODUCTION_TEMPLATE = PRODUCTION_DIR / "base.inp"
PRODUCTION_BASIS = PRODUCTION_DIR / "def2-tzvpd.bas"
LOCAL_BASIS = SCRIPT_DIR / "def2-tzvpd.bas"
RUNS_DIR = SCRIPT_DIR / "runs"
MANIFEST_PATH = SCRIPT_DIR / "manifest.csv"
ORCA_PAL_NPROCS = 16

ATOM_SPECS = [
    {"atom": "H", "charge": 0, "multiplicity": 2},
    {"atom": "O", "charge": 0, "multiplicity": 3},
    {"atom": "N", "charge": 0, "multiplicity": 4},
    {"atom": "S", "charge": 0, "multiplicity": 3},
]

REQUIRED_FRAGMENTS = [
    "! wB97M-V def2-TZVPD EnGrad RIJCOSX def2/J NoUseSym DIIS NOSOSCF NormalConv DEFGRID3 ALLPOP",
    "%pal nprocs 24 end",
    'GTOName "def2-tzvpd.bas"',
    '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end',
    "{{CHARGE}}",
    "{{MULTIPLICITY}}",
    "{{COORDINATES}}",
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def validate_template(text: str) -> None:
    for fragment in REQUIRED_FRAGMENTS:
        if fragment not in text:
            fail(f"Production template is missing required fragment: {fragment}")
    if not text.endswith("\n"):
        fail(f"Production template must end with a trailing newline: {PRODUCTION_TEMPLATE}")


def render_input(template: str, charge: int, multiplicity: int, atom: str) -> str:
    coordinates = f"{atom}   0.00000000   0.00000000   0.00000000"
    rendered = template.replace("%pal nprocs 24 end", f"%pal nprocs {ORCA_PAL_NPROCS} end")
    rendered = re.sub(
        r"\n%loc\s*\n(?:.*\n)*?end\s*\n(?=\*xyz )",
        "\n",
        rendered,
    )
    rendered = rendered.replace("{{CHARGE}}", str(charge))
    rendered = rendered.replace("{{MULTIPLICITY}}", str(multiplicity))
    rendered = rendered.replace("{{COORDINATES}}", coordinates)
    if "{{" in rendered or "}}" in rendered:
        fail(f"Unresolved placeholders remain in rendered ORCA input for {atom}")
    if "%loc" in rendered:
        fail(f"Rendered ORCA input still contains %loc block for {atom}")
    if f"%pal nprocs {ORCA_PAL_NPROCS} end" not in rendered:
        fail(f"Rendered ORCA input has wrong %pal setting for {atom}")
    if f"*xyz {charge} {multiplicity}" not in rendered:
        fail(f"Rendered ORCA input has wrong *xyz header for {atom}")
    return rendered


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    if not PRODUCTION_TEMPLATE.is_file():
        fail(f"Missing production template: {PRODUCTION_TEMPLATE}")
    if not PRODUCTION_BASIS.is_file():
        fail(f"Missing production basis file: {PRODUCTION_BASIS}")

    template = PRODUCTION_TEMPLATE.read_text(encoding="utf-8")
    validate_template(template)
    shutil.copy2(PRODUCTION_BASIS, LOCAL_BASIS)

    manifest_lines = ["atom,charge,multiplicity,run_dir,input_path,output_path"]
    for spec in ATOM_SPECS:
        atom = str(spec["atom"])
        charge = int(spec["charge"])
        multiplicity = int(spec["multiplicity"])
        stem = f"orcaatomization{atom}"
        run_dir = RUNS_DIR / atom
        inp_path = run_dir / f"{stem}.inp"
        out_path = run_dir / f"{stem}.out"
        write_text_lf(inp_path, render_input(template, charge, multiplicity, atom))
        shutil.copy2(LOCAL_BASIS, run_dir / LOCAL_BASIS.name)
        manifest_lines.append(
            f"{atom},{charge},{multiplicity},{run_dir.relative_to(SCRIPT_DIR)},"
            f"{inp_path.relative_to(SCRIPT_DIR)},{out_path.relative_to(SCRIPT_DIR)}"
        )
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")

    write_text_lf(MANIFEST_PATH, "\n".join(manifest_lines) + "\n")
    print(f"wrote {MANIFEST_PATH.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
