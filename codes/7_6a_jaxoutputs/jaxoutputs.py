from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
THERMO_CSV = SCRIPT_DIR.parents[1] / "outputsfull" / "jaxequil" / "thermo.csv"
OUTPUT_DIR = THERMO_CSV.parent / "plots"


def read_thermo_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as handle:
        header = handle.readline().lstrip("#").strip()
        reader = csv.DictReader(handle, fieldnames=[name.strip() for name in header.split(",")])

        rows = list(reader)

    start_index = 0
    for index, row in enumerate(rows):
        if float(row["step"]) == 0.0 and float(row["time"]) == 0.0:
            start_index = index

    data: dict[str, list[float]] = {}
    for row in rows[start_index:]:
        for key, value in row.items():
            if not value:
                continue
            data.setdefault(key, []).append(float(value))

    return data


def save_plot(
    data: dict[str, list[float]],
    y_columns: list[str],
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(9, 5))

    for column in y_columns:
        plt.plot(data["time"], data[column], label=column.replace("_", " ").title())

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(y_columns) > 1:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    data = read_thermo_csv(THERMO_CSV)

    save_plot(
        data,
        ["kinetic_energy", "potential_energy", "total_energy"],
        "Energy",
        "Energy vs Time",
        "energy_vs_time.png",
    )
    save_plot(
        data,
        ["temperature"],
        "Temperature",
        "Temperature vs Time",
        "temperature_vs_time.png",
    )
    save_plot(
        data,
        ["density"],
        "Density",
        "Density vs Time",
        "density_vs_time.png",
    )


if __name__ == "__main__":
    main()
