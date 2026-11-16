#!/usr/bin/env python3
"""Evaluate swappable model/data manifests using the shared parity standard."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from plot_p10_finetuning import ROOT, OUT, sha, validate_frames, replot
from parity_standard import plot_parity


def load_source(record):
    from ase.io import read
    frames, roles = [], []
    model = ROOT / record['model']['path']
    if sha(model) != record['model']['sha256']:
        raise ValueError(f'Model hash mismatch: {model}')
    for dataset in record['datasets']:
        path = ROOT / dataset['path']
        if sha(path) != dataset['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        subset = read(path, ':')
        if len(subset) != dataset['count']:
            raise ValueError(f'Dataset count mismatch: {path}')
        frames.extend(subset)
        roles.extend([dataset['role']] * len(subset))
    validate_frames(frames)
    return model, frames, roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path(__file__).with_name('parity_sources.json'))
    parser.add_argument('--models', nargs='+', default=['s', 'm', 'l'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--check', action='store_true', help='Verify hashes and reference data without inference or output')
    parser.add_argument('--evaluate-only', action='store_true', help='Save reference/prediction arrays without rendering plots')
    parser.add_argument('--from-evaluation', type=Path, help='Redraw saved predictions without inference')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    destination = args.output or OUT / 'visualization' / ('parity_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if args.from_evaluation:
        if args.check or args.evaluate_only:
            parser.error('--check/--evaluate-only cannot be combined with --from-evaluation')
        destination.mkdir(parents=True, exist_ok=False)
        replot(args.from_evaluation, destination)
        print(destination)
        return
    manifest = json.loads(args.sources.read_text())
    # Verify all selected sources before loading any model or creating outputs.
    loaded = [(key, manifest['sources'][key], *load_source(manifest['sources'][key])) for key in args.models]
    for key, record, model, frames, roles in loaded:
        print(f'{key}: {model}; {len(frames)} structures; {record["dtype"]}', flush=True)
    if args.check:
        return
    from mace.calculators import mace_polar
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for key, record, model, frames, roles in loaded:
        folder = destination / key
        folder.mkdir()
        calc = mace_polar(model=str(model), device=args.device, default_dtype=record['dtype'])
        pe, pf = [], []
        for atoms in frames:
            atoms.calc = calc
            pe.append(atoms.get_potential_energy())
            pf.append(atoms.get_forces())
        arrays = dict(reference_energy=np.array([a.info['REF_energy'] for a in frames]),
                      predicted_energy=np.array(pe),
                      reference_forces=np.concatenate([a.arrays['REF_forces'] for a in frames]),
                      predicted_forces=np.concatenate(pf), atom_counts=np.array([len(a) for a in frames]),
                      roles=np.array(roles), atom_symbols=np.array([s for a in frames for s in a.get_chemical_symbols()]))
        metadata = dict(record, model_selection=key, dataset_label=', '.join(sorted(set(roles))),
                        source_manifest=str(args.sources.resolve()), source_manifest_sha256=sha(args.sources))
        if not all(np.isfinite(arrays[k]).all() for k in ('predicted_energy', 'predicted_forces')):
            raise ValueError('Nonfinite MLIP predictions')
        # Persist expensive inference before rendering; redraw needs no model or dataset.
        np.savez_compressed(folder / 'predictions.npz', **arrays)
        (folder / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if not args.evaluate_only:
            plot_parity(folder, *(arrays[k] for k in ('reference_energy', 'predicted_energy', 'reference_forces',
                        'predicted_forces', 'atom_counts', 'roles')), metadata, arrays['atom_symbols'])
        del calc
    print(destination)


if __name__ == '__main__':
    main()
