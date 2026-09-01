#!/usr/bin/env python3
"""Plot per-epoch RMSE from a dated MACE training session as a standalone SVG."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = (
    SCRIPT_DIR
    / "runs"
    / "polar1s_mhft_orca"
    / "logs"
    / "polar1s_mhft_orca_run-3.log"
)

SESSION_START = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}) .*VERIFYING SETTINGS")
EPOCH_METRIC = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) .*Epoch (?P<epoch>\d+): head: "
    r"(?P<head>[^,]+), loss=(?P<loss>[\d.]+), .*RMSE_E_per_atom=\s*(?P<energy>[\d.]+) meV, "
    r"RMSE_F=\s*(?P<force>[\d.]+) meV / A"
)
COLORS = {"Default": "#0072B2", "pt_head": "#D55E00"}


def parse_session(log_path: Path, date: str) -> dict[str, list[tuple[int, float, float, float]]]:
    """Return the final training session started on ``date``.

    A log can contain failed/restarted attempts. Selecting the final
    ``VERIFYING SETTINGS`` block makes this script plot the most recent run
    started that day rather than combining earlier attempts.
    """
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if (match := SESSION_START.match(line)) and match["date"] == date
    ]
    if not starts:
        raise ValueError(f"No training session beginning on {date} in {log_path}")

    start = starts[-1]
    end = next(
        (index for index in range(start + 1, len(lines)) if SESSION_START.match(lines[index])),
        len(lines),
    )
    metrics: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for line in lines[start:end]:
        match = EPOCH_METRIC.match(line)
        # The Aug 31 session continued past midnight.  The session boundary,
        # rather than the calendar date of each metric, identifies its epochs.
        if match:
            metrics[match["head"]].append(
                (
                    int(match["epoch"]),
                    float(match["loss"]),
                    float(match["energy"]),
                    float(match["force"]),
                )
            )
    if not metrics:
        raise ValueError(f"No numeric epoch RMSE values found in the final {date} session")
    return dict(metrics)


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_panel(
    title: str,
    y_label: str,
    values: dict[str, list[tuple[int, float]]],
    panel_top: int,
    width: int,
    panel_height: int,
    *,
    zero_based: bool = True,
    tick_decimals: int = 1,
) -> list[str]:
    left, right, top_margin, bottom = 92, 32, 34, 44
    plot_top = panel_top + top_margin
    plot_width = width - left - right
    plot_height = panel_height - top_margin - bottom
    all_points = [point for points in values.values() for point in points]
    x_min, x_max = min(epoch for epoch, _ in all_points), max(epoch for epoch, _ in all_points)
    y_min, y_max = min(value for _, value in all_points), max(value for _, value in all_points)
    if x_min == x_max:
        x_max += 1
    value_range = y_max - y_min
    scale = max(abs(y_min), abs(y_max), 1.0)
    padding = max(value_range * 0.08, scale * 0.002)
    y_min = max(0.0, y_min - padding) if zero_based else y_min - padding
    y_max += padding

    def x(epoch: int) -> float:
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def y(value: float) -> float:
        return plot_top + (y_max - value) / (y_max - y_min) * plot_height

    svg = [
        f'<text x="{left}" y="{plot_top - 16}" class="title">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_top + plot_height}" class="axis"/>',
        f'<line x1="{left}" y1="{plot_top + plot_height}" x2="{left + plot_width}" y2="{plot_top + plot_height}" class="axis"/>',
        f'<text x="18" y="{plot_top + plot_height / 2}" class="label" transform="rotate(-90 18 {plot_top + plot_height / 2})">{svg_escape(y_label)}</text>',
    ]
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        y_pos = y(value)
        svg += [
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_width}" y2="{y_pos:.1f}" class="grid"/>',
            f'<text x="{left - 8}" y="{y_pos + 4:.1f}" class="tick" text-anchor="end">{value:.{tick_decimals}f}</text>',
        ]
    for step in range(6):
        epoch = round(x_min + step * (x_max - x_min) / 5)
        x_pos = x(epoch)
        svg += [
            f'<line x1="{x_pos:.1f}" y1="{plot_top + plot_height}" x2="{x_pos:.1f}" y2="{plot_top + plot_height + 5}" class="axis"/>',
            f'<text x="{x_pos:.1f}" y="{plot_top + plot_height + 21}" class="tick" text-anchor="middle">{epoch}</text>',
        ]
    svg.append(f'<text x="{left + plot_width / 2:.1f}" y="{plot_top + plot_height + 39}" class="label" text-anchor="middle">Epoch</text>')
    for head, points in sorted(values.items()):
        color = COLORS.get(head, "#009E73")
        point_string = " ".join(f"{x(epoch):.1f},{y(value):.1f}" for epoch, value in points)
        svg.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        if len(values) > 1:
            legend_y = plot_top + 12 + 20 * list(sorted(values)).index(head)
            svg += [
                f'<line x1="{left + plot_width - 138}" y1="{legend_y}" x2="{left + plot_width - 116}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{left + plot_width - 110}" y="{legend_y + 4}" class="tick">{svg_escape(head)}</text>',
            ]
    return svg


def write_svg(metrics: dict[str, list[tuple[int, float, float, float]]], output: Path, date: str) -> None:
    width, header_height, panel_height = 900, 36, 280
    energy = {head: [(epoch, value) for epoch, _, value, _ in points] for head, points in metrics.items()}
    force = {head: [(epoch, value) for epoch, _, _, value in points] for head, points in metrics.items()}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{header_height + panel_height * 2}" viewBox="0 0 {width} {header_height + panel_height * 2}">',
        f'<rect width="{width}" height="{header_height + panel_height * 2}" fill="white"/>',
        "<style>.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font:600 17px sans-serif}.label{font:14px sans-serif}.tick{font:12px sans-serif;fill:#333}</style>",
        f'<text x="{width / 2}" y="24" class="title" text-anchor="middle">MACE RMSE by epoch — final session started {svg_escape(date)}</text>',
    ]
    svg += make_panel("Energy", "RMSE energy per atom (meV)", energy, header_height, width, panel_height)
    svg += make_panel("Forces", "RMSE force (meV / Å)", force, header_height + panel_height, width, panel_height)
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_loss_plot(
    metrics: dict[str, list[tuple[int, float, float, float]]], output: Path, date: str
) -> None:
    """Write a separate two-panel PNG of the evaluated weighted loss by head."""
    width, header_height, panel_height = 900, 36, 280
    loss = {
        head: [(epoch, value) for epoch, value, _, _ in points]
        for head, points in metrics.items()
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{header_height + panel_height * 2}" viewBox="0 0 {width} {header_height + panel_height * 2}">',
        f'<rect width="{width}" height="{header_height + panel_height * 2}" fill="white"/>',
        "<style>.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font:600 17px sans-serif}.label{font:14px sans-serif}.tick{font:12px sans-serif;fill:#333}</style>",
        f'<text x="{width / 2}" y="24" class="title" text-anchor="middle">MACE weighted loss by epoch — final session started {svg_escape(date)}</text>',
    ]
    for index, head in enumerate(sorted(loss)):
        svg += make_panel(
            f"Loss: {head}",
            "Weighted loss",
            {head: loss[head]},
            header_height + index * panel_height,
            width,
            panel_height,
            zero_based=False,
            tick_decimals=4,
        )
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".svg":
        output.write_text("\n".join(svg) + "\n", encoding="utf-8")
        return
    if output.suffix.lower() != ".png":
        raise ValueError("--loss-output must end in .png or .svg")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".svg", encoding="utf-8", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write("\n".join(svg) + "\n")
    try:
        subprocess.run(
            ["rsvg-convert", "--output", str(output), str(temporary_path)],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PNG loss output requires rsvg-convert (librsvg). Use --loss-output "
            "with an .svg suffix when it is unavailable."
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--date", default="2026-08-31", help="Session start date (YYYY-MM-DD).")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "runs" / "polar1s_mhft_orca" / "results" / "rmse_by_epoch_2026-08-31.svg")
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=SCRIPT_DIR / "runs" / "polar1s_mhft_orca" / "results" / "loss_by_epoch_2026-08-31.png",
    )
    args = parser.parse_args()
    metrics = parse_session(args.log, args.date)
    write_svg(metrics, args.output, args.date)
    write_loss_plot(metrics, args.loss_output, args.date)
    print(
        f"Wrote {args.output} and {args.loss_output} with "
        + ", ".join(f"{head}={len(points)} epochs" for head, points in sorted(metrics.items()))
    )


if __name__ == "__main__":
    main()
