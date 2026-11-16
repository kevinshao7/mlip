#!/usr/bin/env python3
"""Render energy-outlier clusters using saved predictions; no model inference."""
import argparse
import json
from pathlib import Path

from plot_p10_finetuning import ROOT, sha, validate_frames


def load_geometry(source, saved, metadata):
    """Require exact dataset identity and reference/atom order before mapping errors."""
    import numpy as np
    from ase.io import read
    frames, roles = [], []
    for dataset in metadata['datasets']:
        path = ROOT / dataset['path']
        if sha(path) != dataset['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        subset = read(path, ':')
        if len(subset) != dataset['count']:
            raise ValueError('Dataset count mismatch')
        frames.extend(subset)
        roles.extend([dataset['role']] * len(subset))
    validate_frames(frames)
    expected = dict(reference_energy=np.array([a.info['REF_energy'] for a in frames]),
                    reference_forces=np.concatenate([a.arrays['REF_forces'] for a in frames]),
                    atom_counts=np.array([len(a) for a in frames]), roles=np.array(roles),
                    atom_symbols=np.array([s for a in frames for s in a.get_chemical_symbols()]))
    for key, array in expected.items():
        if not np.array_equal(saved[key], array):
            raise ValueError(f'Saved {key} disagrees with audited geometry order: {source}')
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path, help='One model folder containing predictions.npz and metrics.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing outputs are preserved')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--highlight-atoms', type=int, default=1, help='Number of largest force-error atoms to ring per energy-outlier cluster')
    args = parser.parse_args()
    if args.frames < 1 or args.fps < 1 or args.highlight_atoms < 1:
        parser.error('Frame, fps and highlighted atom counts must be positive')
    import numpy as np
    from ase.io import write
    from visualize_force_outliers import render_outlier_gif
    with np.load(args.evaluation / 'predictions.npz', allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    metadata = json.loads((args.evaluation / 'metrics.json').read_text())
    frames = load_geometry(args.evaluation, saved, metadata)
    energy_error = 1000 * (saved['predicted_energy'] - saved['reference_energy']) / saved['atom_counts']
    if not np.isfinite(energy_error).all() or not np.isfinite(saved['predicted_forces']).all():
        raise ValueError('Nonfinite saved predictions')
    worst = np.argsort(-np.abs(energy_error), kind='stable')[:5]
    offsets = np.concatenate(([0], np.cumsum(saved['atom_counts'])))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for rank, index in enumerate(worst, 1):
        atoms = frames[index]
        forces = saved['predicted_forces'][offsets[index]:offsets[index + 1]]
        errors = 1000 * np.linalg.norm(forces - atoms.arrays['REF_forces'], axis=1)
        highlighted = np.argsort(-errors, kind='stable')[:args.highlight_atoms]
        pred = atoms.copy()
        pred.arrays['MACE_forces'] = forces
        tag = f'rank{rank:02d}_cluster{index}'
        provenance = {k: str(atoms.info[k]) for k in ('source_file', 'source_frame', 'source_trajectory_id',
                      'source_center_component', 'collection_index') if k in atoms.info}
        records.append(dict(rank=rank, cluster_index=int(index), energy_error_meV_per_atom=float(energy_error[index]),
                            highlighted_atom_indices=highlighted.tolist(), atom_force_errors_meV_per_A=errors.tolist(),
                            source=provenance))
        display = atoms.copy()
        display.arrays['MACE_forces'] = forces
        display.arrays['force_error_meV_per_A'] = errors
        write(args.output / f'{tag}.extxyz', display)
        render_outlier_gif(atoms, pred, int(highlighted[0]),
            f'#{index}; ΔE={energy_error[index]:+.2f} meV/atom', args.output / f'{tag}.gif',
            cutoff=4.5, n_frames=args.frames, fps=args.fps, force_scale=None,
            model_label=metadata['label'], highlight_indices=highlighted)
    (args.output / 'outliers.json').write_text(json.dumps(dict(
        evaluation_directory=str(args.evaluation.resolve()), predictions_sha256=sha(args.evaluation / 'predictions.npz'),
        ranking='absolute total energy error divided by atom count; five per model; stable ties',
        highlight='largest force-vector errors; not an attribution of energy error to atoms',
        index_convention='zero-based cluster/atom indices in saved evaluation order', outliers=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
