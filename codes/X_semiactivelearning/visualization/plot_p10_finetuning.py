#!/usr/bin/env python3
"""Plot the interrupted published P10 run and epoch-9 parity; never train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputsfull/X_semiactivelearning'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputsfull/.cache/matplotlib'))
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_parity(destination, re_, pe, rf, pf, counts, roles, metadata):
    import numpy as np
    import matplotlib.pyplot as plt
    metrics = {}
    for name, cluster_mask in [('all_2000', np.ones(len(counts), dtype=bool)), ('acquisition_1950', np.array(roles) == 'acquisition'), ('held_out_50', np.array(roles) == 'evaluation')]:
        atom_mask = np.repeat(cluster_mask, counts)
        energy_error = (pe[cluster_mask] - re_[cluster_mask]) / counts[cluster_mask]
        force_error = pf[atom_mask] - rf[atom_mask]
        metrics[name] = {'clusters': int(cluster_mask.sum()), 'energy_rmse_meV_per_atom': float(1000*np.sqrt(np.mean(energy_error**2))),
                         'force_component_rmse_meV_per_A': float(1000*np.sqrt(np.mean(force_error**2)))}
    for filename, x, y, label, masks in [
        ('energy_parity.png', re_, pe, 'Absolute electronic energy (eV)', np.array(roles) == 'evaluation'),
        ('force_parity.png', rf.ravel(), pf.ravel(), 'Force component (eV/Å)', np.repeat(np.array(roles) == 'evaluation', counts * 3))]:
        fig, axis = plt.subplots(figsize=(7, 7))
        for mask, label_role in [(~masks, 'Acquisition (includes train/validation)'), (masks, 'Held-out evaluation')]:
            axis.scatter(x[mask], y[mask], s=3, alpha=.3, rasterized=True, label=label_role)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], 'k--', lw=1)
        axis.set(xlabel='DFT ' + label, ylabel='Epoch-9 prediction ' + label,
                 title='P10 epoch 9: all 2000 clusters (mixed splits)', aspect='equal')
        axis.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=220)
        plt.close(fig)
    (destination / 'metrics.json').write_text(json.dumps({**metadata, 'metrics': metrics,
        'note': 'All-data parity is descriptive; correlated acquisition frames are not independent test samples.'}, indent=2) + '\n')


def plot_history(destination, curves):
    import numpy as np
    import matplotlib.pyplot as plt
    np.savetxt(destination / 'validation_history.csv', curves, delimiter=',', header='epoch,loss,energy_RMSE_meV_per_atom,force_RMSE_meV_per_A', comments='')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for column, (axis, label) in enumerate(zip(axes, ['Validation loss', 'Energy RMSE (meV/atom)', 'Force component RMSE (meV/Å)']), 1):
        axis.plot(curves[:, 0], curves[:, column])
        axis.axvline(9, color='orange', ls='--', label='Saved epoch 9')
        axis.set(xlabel='Epoch (−1 = initial)', ylabel=label)
        axis.grid(alpha=.25)
        if column < 3:
            axis.set_yscale('log')
    axes[0].legend()
    fig.suptitle('P10 fine tuning — timed out after epoch 195; validation metrics')
    fig.tight_layout()
    fig.savefig(destination / 'training_performance.png', dpi=200)
    plt.close(fig)


def replot(source, destination):
    import numpy as np
    source = source.resolve()
    with np.load(source / 'predictions.npz', allow_pickle=False) as saved:
        arrays = [saved[key] for key in ('reference_energy', 'predicted_energy',
                  'reference_forces', 'predicted_forces', 'atom_counts', 'roles')]
    metadata = json.loads((source / 'metrics.json').read_text())
    metadata.pop('metrics', None)
    metadata['evaluation_directory'] = str(source)
    plot_parity(destination, *arrays, metadata)
    curves = np.loadtxt(source / 'validation_history.csv', delimiter=',', skiprows=1, ndmin=2)
    plot_history(destination, curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--curves-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--from-evaluation', type=Path, metavar='DIRECTORY',
                        help='Regenerate plots from saved evaluation; no inference')
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone
    destination = args.output or OUT / 'visualization' / ('p10_epoch9_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    if args.from_evaluation:
        replot(args.from_evaluation, destination)
        print(destination)
        return
    log = ROOT / 'outputsfull/slurm/X_P10_polar1l_500ep_833733.out'
    pattern = r'(Initial|Epoch\s+\d+):.*?loss=\s*([\d.eE+-]+), RMSE_E_per_atom=\s*([\d.eE+-]+) meV, RMSE_F=\s*([\d.eE+-]+)'
    rows = [[-1 if epoch == 'Initial' else int(epoch.split()[1]), float(loss), float(energy), float(force)]
            for epoch, loss, energy, force in re.findall(pattern, log.read_text())]
    if not rows:
        raise ValueError('No validation records in training log')
    curves = np.asarray(rows)
    plot_history(destination, curves)
    if args.curves_only:
        print(destination)
        return
    from ase.io import read
    from mace.calculators import mace_polar
    campaign = OUT / 'p10_published_polar1l_500epochs'
    published = json.loads((campaign / 'published_dataset.json').read_text())
    collection = Path(published['collection'])
    summary = json.loads((collection / 'summary.json').read_text())
    # Pin the previously verified epoch-9 export, independently of future P15 pointers.
    model_dir = OUT / 'p15_finetuned_epoch9_1ns_10rep_20261110T102256817401Z/model'
    model = model_dir / 'fine_tuned.model'
    if sha(model) != '481be9d938237e75fa71362adc3601a89e1c4c3626c9ab3094314b09681a43f0':
        raise ValueError('Epoch-9 model hash mismatch')
    frames, roles = [], []
    for role, filename, count in [('acquisition', 'finetuning.xyz', 1950), ('evaluation', 'evaluation.xyz', 50)]:
        path = collection / filename
        if sha(path) != summary['files'][role]['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        subset = read(path, ':')
        if len(subset) != count:
            raise ValueError(f'Expected {count} {role} clusters')
        frames.extend(subset)
        roles.extend([role] * count)
    # Validate every record before any inference; absolute electronic labels only.
    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        if set(symbols) - {'H', 'N', 'O'}:
            raise ValueError('Unsupported formal-charge convention')
        charge = symbols.count('H') - 3 * symbols.count('N') - 2 * symbols.count('O')
        if any(atoms.info.get(k) != charge for k in ('charge', 'orca_input_charge', 'orca_output_charge')):
            raise ValueError('Formal charge mismatch')
        if (any(atoms.info.get(k) != 1 for k in ('spin', 'multiplicity', 'orca_input_multiplicity', 'orca_output_multiplicity'))
                or not atoms.info.get('formal_charge_verified') or (sum(atoms.numbers) - charge) % 2
                or atoms.info.get('energy_reference') != 'absolute'
                or atoms.info.get('dft_functional_basis') != 'wB97M/def2-TZVPD' or atoms.pbc.any()
                or not np.isfinite(atoms.positions).all()
                or not np.isfinite(atoms.info['REF_energy'])
                or not np.isfinite(atoms.arrays['REF_forces']).all()):
            raise ValueError('Invalid reference, geometry, charge or spin provenance')
    calc = mace_polar(model=str(model), device=args.device, default_dtype='float32')
    reference_e, predicted_e, reference_f, predicted_f = [], [], [], []
    for index, atoms in enumerate(frames):
        reference_e.append(atoms.info['REF_energy'])
        reference_f.append(atoms.arrays['REF_forces'].copy())
        atoms.calc = calc
        predicted_e.append(atoms.get_potential_energy())
        predicted_f.append(atoms.get_forces())
        if (index + 1) % 100 == 0:
            print(f'Evaluated {index + 1}/2000', flush=True)
    re_, pe = np.asarray(reference_e), np.asarray(predicted_e)
    rf, pf = np.concatenate(reference_f), np.concatenate(predicted_f)
    if not all(np.isfinite(a).all() for a in (re_, pe, rf, pf)):
        raise ValueError('Nonfinite predictions')
    counts = np.array([len(a) for a in frames])
    np.savez_compressed(destination / 'predictions.npz', reference_energy=re_, predicted_energy=pe,
                        reference_forces=rf, predicted_forces=pf, atom_counts=counts, roles=roles)
    plot_parity(destination, re_, pe, rf, pf, counts, roles,
                {'model': str(model), 'model_sha256': sha(model), 'log': str(log),
                 'status': 'source training timed out; epoch-9 export'})
    print(destination)


if __name__ == '__main__':
    main()
