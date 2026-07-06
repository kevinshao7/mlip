from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
THERMO_CSV = SCRIPT_DIR.parents[1] / "outputsfull" / "run_water_npt" / "thermo.csv"
OUTPUT_PNG = SCRIPT_DIR / "temperature_vs_time.png"


def split_on_time_resets(time, values):
    segments = []
    start = 0
    for index in range(1, len(time)):
        if time[index] < time[index - 1]:
            segments.append((time[start:index], values[start:index]))
            start = index
    segments.append((time[start:], values[start:]))
    return segments


def plot_temperature(thermo_csv: Path = THERMO_CSV, output_png: Path = OUTPUT_PNG) -> Path:
    with thermo_csv.open(newline="") as csv_file:
        first_line = csv_file.readline()
        if first_line.startswith("#"):
            fieldnames = [name.strip() for name in first_line[1:].split(",")]
        else:
            fieldnames = [name.strip() for name in first_line.split(",")]

        reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        rows = list(reader)

    required_columns = ("time", "temperature")
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s) in {thermo_csv}: {missing}")

    time = [float(row["time"]) for row in rows]
    temperature = [float(row["temperature"]) for row in rows]
    last_run_time, last_run_temperature = split_on_time_resets(time, temperature)[-1]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(last_run_time, last_run_temperature, color="#1f77b4", linewidth=1.8)

    ax.set_title("NVT Simulation Temperature - Last Run")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (K)")
    ax.grid(True, alpha=0.3)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_png


if __name__ == "__main__":
    saved_plot = plot_temperature()
    print(f"Saved temperature plot to {saved_plot}")
