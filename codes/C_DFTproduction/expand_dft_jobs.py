#!/usr/bin/env python3
"""Generate BlueHive ORCA input and grouped Slurm files for production cut clusters.
  cd /gpfs/fs2/scratch/kshao4/mlip/codes/C_DFTproduction && for f in expand/C_DFTprod_cutcluster_group_*.slurm; do echo "Submitting $f"; sbatch "$f"; done
Submit generated jobs with:
    for f in expand/*.slurm; do sbatch "$f"; done
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import shlex
import shutil
import sys
from pathlib import Path

from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SLURM = SCRIPT_DIR / "base.slurm"
OUT_DIR = SCRIPT_DIR / "expand"
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTER_XYZ = (
    MLIP_DIR
    / "outputsfull"
    / "C_DFTproduction"
    / "condition_production_dft_clusters.xyz"
)
STEM_PREFIX = "C_DFTprod_cutcluster"
GROUP_PREFIX = "C_DFTprod_cutcluster_group"
MAIL_USER = "ks2120@cam.ac.uk"
DEFAULT_GROUP_SIZE = 10
ORCA_MODULE = "orca/6.1.1"
ORCA_ABSOLUTE_PATH = "/software/orca/6.1.1/orca"
FAIRCHEM_ORCA_CALC = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "calc.py"
FAIRCHEM_SRC = MLIP_DIR / "fairchem" / "src"
FAIRCHEM_ORCA_BASIS = (
    MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "basis" / "def2-tzvpd.bas"
)

FORMAL_CHARGES = {
    "H": 1,
    "N": -3,
    "O": -2,
    "S": -2,
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_frames(spec: str, n_frames: int) -> tuple[int, int]:
    if spec == "all":
        return 0, n_frames
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) != 2:
        fail(f"--frames must be all or start,stop, got {spec!r}")
    start, stop = (int(fields[0]), int(fields[1]))
    if start < 0 or stop <= start:
        fail(f"--frames must satisfy 0 <= start < stop, got {spec!r}")
    if stop > n_frames:
        fail(f"Requested frames {start},{stop}, but cluster XYZ has only {n_frames} frames")
    return start, stop


def stem_for_frame(frame_index: int) -> str:
    return f"{STEM_PREFIX}_{frame_index:04d}"


def group_stem_for_frames(first_frame: int, last_frame: int) -> str:
    return f"{GROUP_PREFIX}_{first_frame:04d}_{last_frame:04d}"


def mail_settings_for_group(group_index: int, n_groups: int) -> str:
    if group_index < 10 or group_index >= n_groups - 10:
        return f"#SBATCH --mail-type=END\n#SBATCH --mail-user={MAIL_USER}"
    return "# Email disabled for middle jobs to avoid notification spam."


def render_stem_list(stems: list[str]) -> str:
    return "\n".join(f'    "{stem}"' for stem in stems)


def formal_charge(symbols: list[str]) -> int:
    charge = 0
    for symbol in symbols:
        try:
            charge += FORMAL_CHARGES[symbol]
        except KeyError as exc:
            raise ValueError(f"No formal charge configured for {symbol!r}") from exc
    return charge


def spin_from_charge(charge: int) -> int:
    return 2 if charge % 2 else 1


def atoms_from_tuples(atoms: list[tuple[str, float, float, float]]) -> Atoms:
    return Atoms(
        symbols=[symbol for symbol, _x, _y, _z in atoms],
        positions=[(x, y, z) for _symbol, x, y, z in atoms],
    )


def parse_comment_metadata(comment: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for token in shlex.split(comment):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metadata[key] = value
    return metadata


def read_xyz_frames(path: Path) -> list[tuple[str, list[tuple[str, float, float, float]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[tuple[str, list[tuple[str, float, float, float]]]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            n_atoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {cursor + 1} in {path}") from exc
        cursor += 1
        if cursor >= len(lines):
            raise ValueError(f"Missing XYZ comment after atom count at line {cursor} in {path}")
        comment = lines[cursor]
        cursor += 1
        if cursor + n_atoms > len(lines):
            raise ValueError(f"Truncated XYZ frame ending after line {cursor + 1} in {path}")

        atoms: list[tuple[str, float, float, float]] = []
        for line in lines[cursor : cursor + n_atoms]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed XYZ atom line: {line!r}")
            atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        frames.append((comment, atoms))
        cursor += n_atoms
    return frames


def load_fairchem_orca_calc():
    if FAIRCHEM_SRC.exists():
        sys.path.insert(0, str(FAIRCHEM_SRC))

    try:
        return importlib.import_module("fairchem.data.omol.orca.calc")
    except ImportError:
        pass

    if not FAIRCHEM_ORCA_CALC.exists():
        return None

    spec = importlib.util.spec_from_file_location("fairchem_omol_orca_calc", FAIRCHEM_ORCA_CALC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load fairchem ORCA calc module from {FAIRCHEM_ORCA_CALC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input_with_fairchem(
    orca_calc,
    atoms: list[tuple[str, float, float, float]],
    charge: int,
    multiplicity: int,
) -> str:
    from ase.calculators.orca import OrcaProfile

    def compatible_orca_profile(command):
        if isinstance(command, list):
            command = command[0] or "orca"
        return OrcaProfile(command)

    orca_calc.OrcaProfile = compatible_orca_profile
    transient_input = OUT_DIR / "orca.inp"
    if transient_input.exists():
        transient_input.unlink()
    orca_calc.write_orca_inputs(
        atoms_from_tuples(atoms),
        OUT_DIR,
        charge=charge,
        mult=multiplicity,
    )
    if not transient_input.is_file():
        fail(f"FairChem did not write expected transient ORCA input: {transient_input}")
    text = transient_input.read_text(encoding="utf-8")
    transient_input.unlink()
    validate_fairchem_input(text, charge, multiplicity)
    return text


def validate_fairchem_input(text: str, charge: int, multiplicity: int) -> None:
    if re.search(r"{{[A-Z_]+}}", text):
        fail("FairChem-generated ORCA input still contains template placeholders")
    stripped_lines = [line.strip() for line in text.splitlines()]
    required_fragments = [
        "! wB97M-V def2-TZVPD",
        "EnGrad",
        "RIJCOSX",
        'GTOName "def2-tzvpd.bas"',
        '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end',
        f"*xyz {charge} {multiplicity}",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            fail(f"FairChem-generated ORCA input is missing expected fragment: {fragment}")
    if not stripped_lines or stripped_lines[-1] != "*":
        fail("FairChem-generated ORCA input must end with the ORCA coordinate terminator '*'")


def render_template(template: str, values: dict[str, str]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    unresolved = sorted(set(re.findall(r"{{[A-Z_]+}}", text)))
    if unresolved:
        fail(f"Unresolved template placeholders: {', '.join(unresolved)}")
    return text


def first_line_number(lines: list[str], pattern: str) -> int:
    regex = re.compile(pattern)
    for index, line in enumerate(lines, start=1):
        if regex.search(line):
            return index
    return 0


def require_line_order(lines: list[str], ordered_patterns: list[tuple[str, str]]) -> None:
    previous_line = 0
    previous_label = "start of file"
    for label, pattern in ordered_patterns:
        line_number = first_line_number(lines, pattern)
        if line_number == 0:
            fail(f"Rendered Slurm is missing required line: {label}")
        if line_number <= previous_line:
            fail(
                f"Rendered Slurm has {label} on line {line_number}, "
                f"before {previous_label} on line {previous_line}"
            )
        previous_line = line_number
        previous_label = label


def parse_rendered_stem_list(slurm_text: str) -> list[str]:
    match = re.search(r"(?ms)^STEMS=\(\n(?P<body>.*?)^\)", slurm_text)
    if not match:
        fail("Rendered Slurm does not contain a STEMS=(...) block")
    stems: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stem_match = re.fullmatch(r'"([^"]+)"', line)
        if not stem_match:
            fail(f"Malformed STEMS entry in rendered Slurm: {raw_line!r}")
        stems.append(stem_match.group(1))
    return stems


def validate_slurm_template(template: str) -> None:
    required_placeholders = {"{{STEM}}", "{{MAIL_SETTINGS}}", "{{STEM_LIST}}"}
    missing = sorted(placeholder for placeholder in required_placeholders if placeholder not in template)
    if missing:
        fail(f"Slurm template is missing placeholders: {', '.join(missing)}")
    forbidden = sorted(set(re.findall(r"{{[A-Z_]+}}", template)) - required_placeholders)
    if forbidden:
        fail(f"Slurm template has unsupported placeholders: {', '.join(forbidden)}")


def validate_rendered_slurm(slurm_text: str, group_stem: str, expected_stems: list[str]) -> None:
    if re.search(r"{{[A-Z_]+}}", slurm_text):
        fail(f"Rendered Slurm for {group_stem} still contains unresolved template placeholders")

    lines = slurm_text.splitlines()
    stripped_lines = [line.strip() for line in lines]
    if not lines or lines[0] != "#!/bin/bash":
        fail(f"Rendered Slurm for {group_stem} must start with #!/bin/bash")
    if f"#SBATCH --job-name={group_stem}" not in lines:
        fail(f"Rendered Slurm for {group_stem} has wrong or missing job name")
    if f"#SBATCH -o {group_stem}.slurmlog.txt" not in lines:
        fail(f"Rendered Slurm for {group_stem} has wrong or missing Slurm stdout path")
    if f"#SBATCH -e {group_stem}.slurmerr.txt" not in lines:
        fail(f"Rendered Slurm for {group_stem} has wrong or missing Slurm stderr path")

    rendered_stems = parse_rendered_stem_list(slurm_text)
    if rendered_stems != expected_stems:
        fail(
            f"Rendered Slurm for {group_stem} has wrong STEMS block: "
            f"expected {expected_stems!r}, got {rendered_stems!r}"
        )

    required_exact_lines = [
        "set -euo pipefail",
        "filter_module_stderr() {",
        'grep -v -F "unalias: sudo: not found" >&2 || true',
        "set +e",
        "module purge 2> >(filter_module_stderr)",
        f"module load {ORCA_MODULE} 2> >(filter_module_stderr)",
        "MODULE_LOAD_STATUS=$?",
        "set -e",
        f'echo "Failed to load {ORCA_MODULE} module" >&2',
        f'ORCA_COMMAND="${{ORCA_COMMAND:-{ORCA_ABSOLUTE_PATH}}}"',
        'if ! command -v mpirun >/dev/null 2>&1; then',
        'echo "mpirun=$(command -v mpirun)"',
        'INPUT_PATH="$MLIP_DIR/codes/C_DFTproduction/expand/${STEM}.inp"',
        'OUTPUT_PATH="$OUTPUT_DIR/${STEM}.out"',
        '"$ORCA_COMMAND" "$INPUT_PATH" > "$OUTPUT_PATH"',
    ]
    for required_line in required_exact_lines:
        if required_line not in stripped_lines:
            fail(f"Rendered Slurm for {group_stem} is missing required line: {required_line}")

    require_line_order(
        lines,
        [
            ("strict bash mode", r"^set -euo pipefail$"),
            ("module stderr filter", r"^filter_module_stderr\(\) \{$"),
            ("disable errexit for module setup", r"^set \+e$"),
            ("module purge", r"^module purge 2> >\(filter_module_stderr\)$"),
            ("ORCA module load", rf"^module load {re.escape(ORCA_MODULE)} 2> >\(filter_module_stderr\)$"),
            ("module load status capture", r"^MODULE_LOAD_STATUS=\$\?$"),
            ("reenable errexit after module setup", r"^set -e$"),
            ("module load failure check", r'^\s*echo "Failed to load orca/6\.1\.1 module" >&2$'),
            ("ORCA command default", rf'^ORCA_COMMAND="\$\{{ORCA_COMMAND:-{re.escape(ORCA_ABSOLUTE_PATH)}\}}"$'),
            ("ORCA executable check", r'^\s*echo "ORCA executable is not available:'),
            ("mpirun availability check", r"^if ! command -v mpirun >/dev/null 2>&1; then$"),
            ("mpirun log line", r'^echo "mpirun=\$\(command -v mpirun\)"$'),
            ("per-STEM loop", r'^for STEM in "\$\{STEMS\[@\]\}"; do$'),
            ("OUTPUT_PATH assignment", r'^\s*OUTPUT_PATH="\$OUTPUT_DIR/\$\{STEM\}\.out"$'),
            ("OUTPUT_PATH completion check", r'^\s*if \[\[ -f "\$OUTPUT_PATH" \]\]'),
            ("ORCA run command", r'^\s*"\$ORCA_COMMAND" "\$INPUT_PATH" > "\$OUTPUT_PATH"$'),
        ],
    )


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def remove_stale_generated() -> None:
    for pattern in (f"{STEM_PREFIX}_*.inp", f"{STEM_PREFIX}_*.slurm", f"{GROUP_PREFIX}_*.slurm"):
        for path in OUT_DIR.glob(pattern):
            path.unlink()
    manifest = OUT_DIR / "manifest.csv"
    if manifest.exists():
        manifest.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="all", help="Frame range: all or half-open start,stop")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_XYZ)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--clean", action="store_true", help="Remove stale generated files in expand/ first.")
    args = parser.parse_args()

    if not BASE_SLURM.is_file():
        fail(f"Missing Slurm template: {BASE_SLURM}")
    if not args.clusters.is_file():
        fail(f"Missing cluster XYZ: {args.clusters}")
    if args.group_size < 1:
        fail("--group-size must be >= 1")

    orca_calc = load_fairchem_orca_calc()
    if orca_calc is None:
        fail(
            "Could not import fairchem.data.omol.orca.calc and could not load it from "
            f"{FAIRCHEM_ORCA_CALC}"
        )

    frames = read_xyz_frames(args.clusters)
    start, stop = parse_frames(args.frames, len(frames))

    OUT_DIR.mkdir(exist_ok=True)
    if args.clean:
        remove_stale_generated()

    slurm_template = BASE_SLURM.read_text(encoding="utf-8")
    validate_slurm_template(slurm_template)
    if FAIRCHEM_ORCA_BASIS.exists():
        shutil.copy2(FAIRCHEM_ORCA_BASIS, SCRIPT_DIR / FAIRCHEM_ORCA_BASIS.name)

    manifest_lines = [
        "frame,stem,input,slurm,charge,multiplicity,n_atoms,sample_kind,source_condensed_frame,sample_order"
    ]
    frame_records: list[dict[str, object]] = []
    for frame_index in range(start, stop):
        comment, atoms = frames[frame_index]
        metadata = parse_comment_metadata(comment)
        stem = stem_for_frame(frame_index)
        symbols = [symbol for symbol, _x, _y, _z in atoms]
        charge = int(metadata.get("charge", formal_charge(symbols)))
        multiplicity = int(metadata.get("spin", spin_from_charge(charge)))

        inp_path = OUT_DIR / f"{stem}.inp"
        write_text_lf(inp_path, make_input_with_fairchem(orca_calc, atoms, charge, multiplicity))
        frame_records.append(
            {
                "frame_index": frame_index,
                "stem": stem,
                "input": f"expand/{inp_path.name}",
                "charge": charge,
                "multiplicity": multiplicity,
                "n_atoms": len(atoms),
                "sample_kind": metadata.get("sample_kind", ""),
                "source_condensed_frame": metadata.get("source_condensed_frame", ""),
                "sample_order": metadata.get("sample_order", ""),
            }
        )
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")

    groups = [
        frame_records[index : index + args.group_size]
        for index in range(0, len(frame_records), args.group_size)
    ]
    slurm_by_stem: dict[str, str] = {}
    for group_index, group in enumerate(groups):
        first_frame = int(group[0]["frame_index"])
        last_frame = int(group[-1]["frame_index"])
        group_stem = group_stem_for_frames(first_frame, last_frame)
        group_stems = [str(record["stem"]) for record in group]
        values = {
            "STEM": group_stem,
            "MAIL_SETTINGS": mail_settings_for_group(group_index, len(groups)),
            "STEM_LIST": render_stem_list(group_stems),
        }
        slurm_path = OUT_DIR / f"{group_stem}.slurm"
        slurm_text = render_template(slurm_template, values)
        validate_rendered_slurm(slurm_text, group_stem, group_stems)
        write_text_lf(slurm_path, slurm_text)
        for stem in group_stems:
            slurm_by_stem[stem] = f"expand/{slurm_path.name}"
        print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)} for {len(group)} ORCA input(s)")

    for record in frame_records:
        stem = str(record["stem"])
        manifest_lines.append(
            f"{record['frame_index']},{stem},{record['input']},{slurm_by_stem[stem]},"
            f"{record['charge']},{record['multiplicity']},{record['n_atoms']},"
            f"{record['sample_kind']},{record['source_condensed_frame']},{record['sample_order']}"
        )

    write_text_lf(OUT_DIR / "manifest.csv", "\n".join(manifest_lines) + "\n")
    print(
        f"Generated {stop - start} input files and {len(groups)} grouped Slurm files "
        f"in {OUT_DIR.name}/; group size {args.group_size}"
    )


if __name__ == "__main__":
    main()
