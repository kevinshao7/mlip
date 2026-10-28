#!/usr/bin/env python3
"""Create GIFs of the worst force outlier atoms for foundation and fine-tuned models.

For each outlier atom, renders a rotating 3D view of the local cluster environment
with the DFT force vector (blue) and MLIP force vector (red) drawn on the outlier atom.
Outliers are ranked by |F_MLIP - F_DFT| per atom, matching the force_parity.png plot.

Usage (from X_semiactivelearning/):
    /home/kevinsh/env/bin/python visualize_force_outliers.py
    /home/kevinsh/env/bin/python visualize_force_outliers.py --dataset all --top 5
    /home/kevinsh/env/bin/python visualize_force_outliers.py --cutoff 5.0 --frames 72
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_MPL_CACHE = Path(__file__).resolve().parent / "runs" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

CODES = Path(__file__).resolve().parent
MLIP = CODES.parents[1]
OUT  = MLIP / "outputsfull" / "X_semiactivelearning"

DATASETS = {
    "p31_r2": OUT / "cycles" / "P31p6228GPa" / "dft" / "acquisition_r2" / "processed.extxyz",
    "all":    OUT / "cycles" / "P31p6228GPa" / "training" / "cumulative_all.extxyz",
    "val":    OUT / "cycles" / "P31p6228GPa" / "training" / "random_validation.extxyz",
}

# Atom display colours (element → hex)
ATOM_COLORS = {"H": "#AAAAAA", "N": "#3050F8", "O": "#FF0D0D"}
ATOM_RADII  = {"H": 0.31, "N": 0.71, "O": 0.66}   # covalent radii in Å (for marker size)
DEFAULT_COLOR  = "#888888"
DEFAULT_RADIUS = 0.5

DFT_COLOR      = "#0072B2"   # blue — DFT force arrow
MLIP_COLOR     = "#D55E00"   # vermillion — MLIP force arrow
OUTLIER_COLOR  = "#FFD700"   # gold ring on the outlier atom
ERROR_COLOR    = "#009E73"   # green — error vector (MLIP - DFT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), default="all")
    p.add_argument("--top",     type=int, default=5, help="Number of worst outliers per model")
    p.add_argument("--cutoff",  type=float, default=4.5,
                   help="Show atoms within this distance (Å) of the outlier (default 4.5)")
    p.add_argument("--frames",  type=int, default=60,
                   help="Number of rotation frames in the GIF (default 60)")
    p.add_argument("--fps",     type=int, default=10,
                   help="Frames per second (default 10 → 6 s for 60 frames)")
    p.add_argument("--force-scale", type=float, default=None,
                   help="Å per meV/Å for arrow length (auto-scaled if omitted)")
    return p.parse_args()


def neighbour_indices(pos, centre_idx: int, cutoff: float, cell=None):
    """Return indices of atoms within `cutoff` Å of `pos[centre_idx]`, PBC-aware if cell is valid."""
    import numpy as np
    anchor = pos[centre_idx]
    delta = pos - anchor
    if cell is not None:
        # Only apply PBC if the cell is non-singular (i.e. a real periodic box)
        try:
            if np.linalg.matrix_rank(cell) == 3:
                inv = np.linalg.inv(cell.T)
                frac = delta @ inv
                frac -= np.round(frac)
                delta = frac @ cell.T
        except np.linalg.LinAlgError:
            pass  # fall back to plain Cartesian
    dist = np.linalg.norm(delta, axis=1)
    return np.where((dist < cutoff) & (dist > 0))[0], delta


def make_gif(
    frames_rgb: list,
    path: Path,
    fps: int,
) -> None:
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames_rgb]
    duration_ms = max(1, 1000 // fps)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0)


def render_outlier_gif(
    atoms_dft,       # ASE Atoms for the config (DFT reference)
    atoms_pred,      # ASE Atoms with MACE_forces
    centre_idx: int,
    label: str,
    output_path: Path,
    cutoff: float,
    n_frames: int,
    fps: int,
    force_scale: float | None,
    model_label: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    pos  = atoms_dft.get_positions()
    cell = atoms_dft.get_cell()
    syms = atoms_dft.get_chemical_symbols()

    F_dft  = atoms_dft.arrays["REF_forces"][centre_idx]   # eV/Å
    F_mlip = atoms_pred.arrays["MACE_forces"][centre_idx]  # eV/Å
    F_err  = F_mlip - F_dft

    # Convert to meV/Å for display
    F_dft_mev  = F_dft  * 1000.0
    F_mlip_mev = F_mlip * 1000.0
    F_err_mev  = F_err  * 1000.0

    # Show the entire cluster, centred on the outlier atom with minimum-image wrapping
    anchor = pos[centre_idx]
    delta  = pos - anchor
    cell_mat = np.array(cell)
    if np.linalg.matrix_rank(cell_mat) == 3:
        try:
            inv  = np.linalg.inv(cell_mat.T)
            frac = delta @ inv
            frac -= np.round(frac)          # wrap to [-0.5, 0.5) in fractional coords
            delta = frac @ cell_mat.T
        except np.linalg.LinAlgError:
            pass   # non-periodic cluster — use raw Cartesian delta
    show_pos  = delta                   # outlier atom at origin
    show_idx  = list(range(len(pos)))
    show_syms = syms

    # Arrow scale: auto so the longest force fills ~25% of the cluster extent
    all_force_mags = [np.linalg.norm(F_dft_mev), np.linalg.norm(F_mlip_mev)]
    max_mag = max(all_force_mags) if max(all_force_mags) > 0 else 1.0
    view_radius = max(np.linalg.norm(show_pos, axis=1).max(), 1.0)
    if force_scale is None:
        scale = 0.35 * view_radius / max_mag
    else:
        scale = force_scale

    marker_sizes = [max(80, 4000 * ATOM_RADII.get(s, DEFAULT_RADIUS) ** 2) for s in show_syms]
    colors       = [ATOM_COLORS.get(s, DEFAULT_COLOR) for s in show_syms]

    dft_err_mag  = float(np.linalg.norm(F_err_mev))
    dft_mag_val  = float(np.linalg.norm(F_dft_mev))
    mlip_mag_val = float(np.linalg.norm(F_mlip_mev))

    title = (f"{model_label}  |  atom {centre_idx} ({syms[centre_idx]})  "
             f"config {label}\n"
             f"|F_DFT|={dft_mag_val:.1f}  |F_MLIP|={mlip_mag_val:.1f}  "
             f"|ΔF|={dft_err_mag:.1f}  meV/Å")

    lim = max(view_radius * 1.3, np.linalg.norm(F_dft_mev) * scale * 1.2,
              np.linalg.norm(F_mlip_mev) * scale * 1.2)

    rgb_frames = []
    for angle in np.linspace(0, 360, n_frames, endpoint=False):
        fig = plt.figure(figsize=(7, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        # Atoms
        xs, ys, zs = show_pos[:, 0], show_pos[:, 1], show_pos[:, 2]
        # All atoms except the outlier
        other = [i for i in range(len(show_pos)) if i != centre_idx]
        if other:
            ax.scatter(xs[other], ys[other], zs[other],
                       s=[marker_sizes[i] for i in other],
                       c=[colors[i] for i in other],
                       depthshade=True, alpha=0.75, edgecolors="none")
        # Outlier atom with gold ring (centred at origin)
        ax.scatter([show_pos[centre_idx, 0]], [show_pos[centre_idx, 1]], [show_pos[centre_idx, 2]],
                   s=marker_sizes[centre_idx], c=colors[centre_idx],
                   edgecolors=OUTLIER_COLOR, linewidths=2.5, depthshade=False, zorder=5)

        # Force arrows originate from the outlier atom (always at origin after wrapping)
        ox, oy, oz = show_pos[centre_idx]   # should be (0,0,0) but use directly for safety
        def arrow(vec, color, lbl):
            v = vec * scale
            ax.quiver(ox, oy, oz, v[0], v[1], v[2],
                      color=color, arrow_length_ratio=0.2, linewidth=2.0,
                      label=f"{lbl} ({np.linalg.norm(vec):.1f} meV/Å)")

        arrow(F_dft_mev,  DFT_COLOR,  "DFT")
        arrow(F_mlip_mev, MLIP_COLOR, "MLIP")
        arrow(F_err_mev,  ERROR_COLOR, "Error (MLIP−DFT)")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("x (Å)", fontsize=7)
        ax.set_ylabel("y (Å)", fontsize=7)
        ax.set_zlabel("z (Å)", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.view_init(elev=20, azim=angle)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        from PIL import Image
        img = Image.open(buf).convert("RGB")
        rgb_frames.append(img)
        buf.close()

    # Save as GIF (PIL list of Images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, 1000 // fps)
    rgb_frames[0].save(
        output_path, save_all=True, append_images=rgb_frames[1:],
        duration=duration_ms, loop=0,
    )
    print(f"  → {output_path}")


def main() -> None:
    import numpy as np
    from ase.io import read as ase_read

    args = parse_args()
    configs_path = DATASETS[args.dataset]
    comp_dir = OUT / "foundation_comparison" / args.dataset

    foundation_pred_path = comp_dir / "foundation_predictions.xyz"
    finetuned_pred_path  = comp_dir / "finetuned_predictions.xyz"

    for p in (configs_path, foundation_pred_path, finetuned_pred_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing file: {p}\nRun compare_foundation.py --dataset {args.dataset} first."
            )

    print(f"Loading structures from {configs_path} ...")
    dft_frames = ase_read(str(configs_path), index=":")
    fnd_frames = ase_read(str(foundation_pred_path), index=":")
    tnd_frames = ase_read(str(finetuned_pred_path), index=":")

    n = len(dft_frames)
    print(f"  {n} configs loaded")

    # Build per-atom arrays: [N_atoms, 3] force vectors and element list
    dft_F_list, fnd_F_list, tnd_F_list = [], [], []
    atom_config_idx, atom_idx_in_config, elem_list = [], [], []

    for ci, (dft, fnd, tnd) in enumerate(zip(dft_frames, fnd_frames, tnd_frames)):
        n_at = len(dft)
        dft_F_list.append(dft.arrays["REF_forces"])
        fnd_F_list.append(fnd.arrays["MACE_forces"])
        tnd_F_list.append(tnd.arrays["MACE_forces"])
        atom_config_idx.extend([ci] * n_at)
        atom_idx_in_config.extend(range(n_at))
        elem_list.extend(dft.get_chemical_symbols())

    dft_Fvec = np.concatenate(dft_F_list, axis=0)   # [N, 3]
    fnd_Fvec = np.concatenate(fnd_F_list, axis=0)
    tnd_Fvec = np.concatenate(tnd_F_list, axis=0)

    fnd_dF = np.linalg.norm(fnd_Fvec - dft_Fvec, axis=1) * 1000.0  # meV/Å
    tnd_dF = np.linalg.norm(tnd_Fvec - dft_Fvec, axis=1) * 1000.0

    atom_config_idx     = np.array(atom_config_idx)
    atom_idx_in_config  = np.array(atom_idx_in_config)

    out_gif_dir = comp_dir / "outlier_gifs"
    out_gif_dir.mkdir(parents=True, exist_ok=True)

    for model_tag, err_arr, pred_frames in [
        ("foundation", fnd_dF, fnd_frames),
        ("finetuned",  tnd_dF, tnd_frames),
    ]:
        worst = np.argsort(err_arr)[-args.top:][::-1]
        print(f"\n{'='*60}")
        print(f"{model_tag.upper()}  — top {args.top} worst atoms by |ΔF|")
        print(f"{'='*60}")

        for rank, flat_idx in enumerate(worst, start=1):
            ci   = int(atom_config_idx[flat_idx])
            ai   = int(atom_idx_in_config[flat_idx])
            elem = elem_list[flat_idx]
            err  = float(err_arr[flat_idx])

            config_label = dft_frames[ci].info.get("source_file", f"config{ci}")
            if isinstance(config_label, str):
                config_label = Path(config_label).stem

            print(f"  rank {rank}: config={ci} atom={ai} ({elem})  |ΔF|={err:.1f} meV/Å  src={config_label}")

            gif_name = f"{model_tag}_rank{rank:02d}_cfg{ci}_atom{ai}_{elem}.gif"
            render_outlier_gif(
                atoms_dft   = dft_frames[ci],
                atoms_pred  = pred_frames[ci],
                centre_idx  = ai,
                label       = config_label,
                output_path = out_gif_dir / gif_name,
                cutoff      = args.cutoff,
                n_frames    = args.frames,
                fps         = args.fps,
                force_scale = args.force_scale,
                model_label = model_tag,
            )

    print(f"\nAll GIFs written to {out_gif_dir}/")


if __name__ == "__main__":
    main()
