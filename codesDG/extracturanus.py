"""Extract Uranus pressure-density-temperature profiles from Table 4 text.

The local ``uranus.txt`` file is a tab-separated copy of Table 4 from
Scheibe et al., MNRAS 487, 2653 (2019). It contains one pressure and density
profile plus three temperature/composition profile cases. This script writes
one wide CSV with shared pressure/density columns and one temperature column
per case.

https://academic.oup.com/mnras/article/487/2/2653/5505844?login=true
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TEMPERATURE_COLUMNS = {
    "hot_uranus_temperature_K": 2,
    "cold_uranus_temperature_K": 4,
    "preferred_uranus_temperature_K": 6,
}


def parse_table_number(text: str) -> float:
    """Parse table numbers, including unicode scientific notation."""
    cleaned = (
        text.strip()
        .replace("\u2060", "")
        .replace("\u2212", "-")
        .replace("\u00d7", "x")
    )
    if "x" in cleaned:
        mantissa, exponent = cleaned.split("x", maxsplit=1)
        exponent = exponent.strip()
        if exponent.startswith("10"):
            exponent = exponent[2:]
        return float(mantissa.strip()) * 10.0 ** float(exponent.strip())
    return float(cleaned)


def read_uranus_profiles(table_path: Path) -> list[dict[str, object]]:
    """Return wide P-rho-T rows for the three Table 4 cases.

    Density is recorded as g/cm^3 in the paper table.  Repeated pressures are
    preserved because they mark discontinuities between Uranus layers.
    """
    rows: list[dict[str, object]] = []

    with table_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)  # header copied from the paper table

        for source_row, fields in enumerate(reader, start=2):
            if not fields or not fields[0].strip():
                continue

            pressure_gpa = parse_table_number(fields[0])
            density_g_cm3 = parse_table_number(fields[1])
            row: dict[str, object] = {
                "source_row": source_row,
                "pressure_GPa": pressure_gpa,
                "density_g_cm3": density_g_cm3,
            }
            for column_name, temperature_col in TEMPERATURE_COLUMNS.items():
                row[column_name] = parse_table_number(fields[temperature_col])
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_row",
        "pressure_GPa",
        "density_g_cm3",
        *TEMPERATURE_COLUMNS.keys(),
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(format_csv_row(row) for row in rows)


def format_csv_row(row: dict[str, object]) -> dict[str, object]:
    """Format floats compactly while preserving numeric CSV values."""
    formatted: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.12g}"
        else:
            formatted[key] = value
    return formatted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract P-rho-T profiles for the three Uranus Table 4 cases."
    )
    default_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "uranus.txt",
        help="Path to the tab-separated Table 4 text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "uranus_profiles.csv",
        help="Wide CSV output containing all three temperature cases.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = read_uranus_profiles(args.input)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} Uranus profile rows to {args.output}")


if __name__ == "__main__":
    main()
