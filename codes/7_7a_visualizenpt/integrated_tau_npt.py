from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


AMU_TO_G = 1.66053906660e-24
ANG3_TO_CM3 = 1e-24
MASSES = {"H": 1.00784, "C": 12.011, "N": 14.0067, "O": 15.999, "S": 32.06}
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


@dataclass
class TauResult:
    name: str
    unit: str
    mean: float
    observable_variance: float
    standard_error: float
    tau_int_samples: float
    tau_int_ps: float
    effective_sample_size: float
    selected_window: int
    converged: bool
    n_samples: int
    dt_ps: float
    warnings: list[str]


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files[0] if files else None


def find_thermo(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    txt = find_one(input_path, "*thermo*.txt") or find_one(input_path, "*.txt")
    if txt is None:
        raise FileNotFoundError(f"No thermo text file found in {input_path}")
    return txt


def load_thermo(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[0].lstrip("#").split()
    rows = [[float(x) for x in line.split()] for line in lines[1:] if line.strip()]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    if len(rows[0]) != len(header):
        header = [
            "time_fs",
            "temperature_K",
            "pressure_GPa",
            "energy_eV_per_atom",
            "kinetic_energy_eV_per_atom",
            "total_energy_eV_per_atom",
        ][: len(rows[0])]
    return {key: [row[i] for row in rows] for i, key in enumerate(header)}


def det3(cell: list[float]) -> float:
    a, b, c, d, e, f, g, h, i = cell
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def xyz_densities(path: Path) -> list[float]:
    densities: list[float] = []
    total_mass: float | None = None
    with path.open(encoding="utf-8", errors="ignore") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line in {path}: {line!r}") from exc

            comment = fh.readline()
            match = LATTICE_RE.search(comment)
            symbols = []
            for _ in range(natoms):
                atom_line = fh.readline()
                if not atom_line:
                    raise ValueError(f"Unexpected end of file while reading {path}")
                if total_mass is None:
                    symbols.append(atom_line.split()[0])

            if total_mass is None:
                try:
                    total_mass = sum(MASSES[symbol] for symbol in symbols)
                except KeyError as exc:
                    raise KeyError(f"No mass configured for element {exc.args[0]!r}") from exc

            if not match:
                densities.append(float("nan"))
                continue
            cell = [float(x) for x in match.group(1).split()]
            if len(cell) != 9:
                raise ValueError(f"Expected 9 lattice entries in {path}, got {len(cell)}")
            volume_ang3 = abs(det3(cell))
            densities.append(total_mass * AMU_TO_G / (volume_ang3 * ANG3_TO_CM3))
    return densities


def finite_pairs(times: list[float], values: list[float], cutoff_ps: float) -> tuple[list[float], list[float]]:
    pairs = [
        (time_fs / 1000.0, value)
        for time_fs, value in zip(times, values)
        if time_fs / 1000.0 >= cutoff_ps and math.isfinite(value)
    ]
    if not pairs:
        raise ValueError(f"Cutoff {cutoff_ps:g} ps leaves no samples")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def integrated_tau(
    name: str,
    unit: str,
    times_ps: list[float],
    values: list[float],
    max_lag: int | None,
    window_factor: float,
) -> TauResult:
    n = len(values)
    warnings: list[str] = []
    if n < 3:
        raise ValueError(f"{name} needs at least 3 samples after cutoff")

    diffs = [times_ps[i + 1] - times_ps[i] for i in range(n - 1)]
    dt_ps = mean(diffs)
    if any(abs(d - dt_ps) > max(1e-12, abs(dt_ps) * 1e-6) for d in diffs):
        warnings.append("Sampling interval is not perfectly constant; using mean dt.")

    avg = mean(values)
    centered = [value - avg for value in values]
    c0 = math.fsum(x * x for x in centered) / n
    if c0 <= 0.0 or not math.isfinite(c0):
        raise ValueError(f"{name} has non-positive variance")

    lag_limit = n - 1 if max_lag is None else min(max_lag, n - 1)
    tau = 0.5
    selected_window = lag_limit
    converged = False

    for lag in range(1, lag_limit + 1):
        cov = math.fsum(centered[i] * centered[i + lag] for i in range(n - lag)) / (n - lag)
        rho = cov / c0
        tau += rho
        if math.isfinite(tau) and tau > 0.0 and lag >= window_factor * tau:
            selected_window = lag
            converged = True
            break

    if not math.isfinite(tau) or tau <= 0.0:
        warnings.append("Integrated tau estimate is non-positive or non-finite; SEM is unreliable.")
        standard_error = float("nan")
        effective_n = float("nan")
    else:
        effective_n = n / (2.0 * tau)
        standard_error = math.sqrt(2.0 * tau * c0 / n)

    if not converged:
        warnings.append(
            f"No self-consistent window M >= {window_factor:g}*tau_int(M) found up to lag {lag_limit}."
        )
    if converged and n < 50.0 * tau:
        warnings.append("Production length is less than 50 tau_int; uncertainty estimate may be noisy.")

    return TauResult(
        name=name,
        unit=unit,
        mean=avg,
        observable_variance=c0,
        standard_error=standard_error,
        tau_int_samples=tau,
        tau_int_ps=tau * dt_ps,
        effective_sample_size=effective_n,
        selected_window=selected_window,
        converged=converged,
        n_samples=n,
        dt_ps=dt_ps,
        warnings=warnings,
    )


def write_summary(path: Path, results: list[TauResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "observable",
                "unit",
                "mean",
                "standard_error",
                "tau_int_samples",
                "tau_int_ps",
                "effective_sample_size",
                "selected_window",
                "observable_variance",
                "n_samples",
                "dt_ps",
                "converged",
                "warnings",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.unit,
                    f"{result.mean:.12g}",
                    f"{result.standard_error:.12g}",
                    f"{result.tau_int_samples:.12g}",
                    f"{result.tau_int_ps:.12g}",
                    f"{result.effective_sample_size:.12g}",
                    result.selected_window,
                    f"{result.observable_variance:.12g}",
                    result.n_samples,
                    f"{result.dt_ps:.12g}",
                    result.converged,
                    "; ".join(result.warnings),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate NPT temperature, pressure, and density means with integrated-tau errors."
    )
    parser.add_argument("input", type=Path, help="Run folder or thermo text file.")
    parser.add_argument("--cutoff-ps", type=float, default=5.0, help="Equilibration cutoff in ps.")
    parser.add_argument("--max-lag", type=int, default=None, help="Maximum lag in samples; default N-1.")
    parser.add_argument("--window-factor", type=float, default=10.0, help="Self-consistent window factor.")
    args = parser.parse_args()

    thermo_path = find_thermo(args.input)
    run_dir = thermo_path.parent
    data = load_thermo(thermo_path)
    xyz_path = find_one(run_dir, f"{thermo_path.stem.replace('_thermo', '')}*.xyz") or find_one(run_dir, "*.xyz")
    if xyz_path is None:
        raise FileNotFoundError(f"No XYZ trajectory found in {run_dir}; density cannot be computed")
    densities = xyz_densities(xyz_path)

    n = min(len(data["time_fs"]), len(densities))
    times = data["time_fs"][:n]
    observables = [
        ("temperature", "K", data["temperature_K"][:n]),
        ("pressure", "GPa", data["pressure_GPa"][:n]),
        ("density", "g/cm^3", densities[:n]),
    ]

    results = []
    for name, unit, values in observables:
        prod_times, prod_values = finite_pairs(times, values, args.cutoff_ps)
        results.append(integrated_tau(name, unit, prod_times, prod_values, args.max_lag, args.window_factor))

    out = run_dir / f"{thermo_path.stem}_integrated_tau_after_{args.cutoff_ps:g}ps.csv"
    write_summary(out, results)

    print(f"thermo: {thermo_path}")
    print(f"xyz: {xyz_path}")
    print(f"production cutoff: {args.cutoff_ps:g} ps")
    print(f"summary: {out}")
    for result in results:
        pm = "+/-"
        print(
            f"{result.name:11s} {result.mean:.8g} {pm} {result.standard_error:.3g} {result.unit}; "
            f"tau_int={result.tau_int_samples:.4g} samples ({result.tau_int_ps:.4g} ps), "
            f"N_eff={result.effective_sample_size:.4g}, M={result.selected_window}, "
            f"converged={result.converged}"
        )
        for warning in result.warnings:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
