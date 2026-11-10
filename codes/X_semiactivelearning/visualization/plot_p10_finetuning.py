#!/usr/bin/env python3
"""Plot P10 foundation or epoch-9 parity and saved evaluations; never train."""
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


def resolve_model(selection):
    if selection == 'finet':
        model = OUT / 'p15_finetuned_epoch9_1ns_10rep_20261110T102256817401Z/model/fine_tuned.model'
        expected = '481be9d938237e75fa71362adc3601a89e1c4c3626c9ab3094314b09681a43f0'
    else:
        workflow = Path(__file__).resolve().parents[1]
        pointer = json.loads((workflow / 'partial_clusters_active.json').read_text())
        plan = json.loads((ROOT / pointer['source_plan']).read_text())
        record = json.loads((ROOT / plan['output_relative'] / 'model_binding.json').read_text())
        if plan.get('model_mode') != 'foundation' or record.get('model_mode') != 'foundation' or record.get('foundation_model') != 'polar-1-l':
            raise ValueError('Dataset source binding must select foundation polar-1-l')
        expected = plan['foundation_checkpoint']['sha256']
        if record['sha256'] != expected:
            raise ValueError('Foundation binding disagrees with checkpoint lock')
        model = ROOT / record['model_relative']
        if sha(ROOT / record['provenance_relative']) != record['provenance_sha256']:
            raise ValueError('Foundation provenance hash mismatch')
    if sha(model) != expected:
        raise ValueError(f'{selection} model hash mismatch')
    return model


def plot_parity(destination, re_, pe, rf, pf, counts, roles, metadata, atom_symbols):
    import numpy as np
    import matplotlib.pyplot as plt
    model_label = 'Foundation polar-1-l' if metadata.get('model_selection') == 'found' else 'Fine-tuned epoch 9'
    metrics = {}
    for name, cluster_mask in [('all_2000', np.ones(len(counts), dtype=bool)), ('acquisition_1950', np.array(roles) == 'acquisition'), ('held_out_50', np.array(roles) == 'evaluation')]:
        atom_mask = np.repeat(cluster_mask, counts)
        energy_error = (pe[cluster_mask] - re_[cluster_mask]) / counts[cluster_mask]
        force_error = pf[atom_mask] - rf[atom_mask]
        metrics[name] = {'clusters': int(cluster_mask.sum()), 'energy_rmse_meV_per_atom': float(1000*np.sqrt(np.mean(energy_error**2))),
                         'force_component_rmse_meV_per_A': float(1000*np.sqrt(np.mean(force_error**2)))}
    atom_symbols = np.asarray(atom_symbols)
    if atom_symbols.shape != (len(rf),) or not set(atom_symbols) <= {'H', 'N', 'O'}:
        raise ValueError('Expected one supported atom symbol per force vector')
    fig, axis = plt.subplots(figsize=(7, 7))
    held_out = np.array(roles) == 'evaluation'
    for mask, label in [(~held_out, 'Acquisition (includes train/validation)'), (held_out, 'Held-out evaluation')]:
        axis.scatter(1000 * re_[mask], 1000 * (pe[mask] - re_[mask]),
                     s=3, alpha=.3, rasterized=True, label=label)
    energy_errors = 1000 * (pe - re_)
    axis.text(.03, .97,
              f'All clusters: RMSE = {np.sqrt(np.mean(energy_errors**2)):.2f} meV\n'
              f'MAE = {np.mean(np.abs(energy_errors)):.2f} meV',
              transform=axis.transAxes, va='top', fontsize=9,
              bbox=dict(facecolor='white', alpha=.85, edgecolor='.8'))
    axis.axhline(0, color='k', ls='--', lw=1)
    axis.set(xlabel='DFT absolute electronic energy (meV)',
             ylabel='MLIP − DFT energy (meV)',
             title=f'P10 {model_label}: all {len(counts)} clusters (mixed splits)')
    axis.legend(markerscale=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(destination / 'energy_parity.png', dpi=220)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 7))
    dft_magnitude = 1000 * np.linalg.norm(rf, axis=1)
    error_magnitude = 1000 * np.linalg.norm(pf - rf, axis=1)
    force_statistics = [f'All atoms: RMSE = {np.sqrt(np.mean(error_magnitude**2)):.2f}, MAE = {np.mean(error_magnitude):.2f}']
    for symbol, color in [('H', 'tab:blue'), ('N', 'tab:green'), ('O', 'tab:red')]:
        mask = atom_symbols == symbol
        if mask.any():
            force_statistics.append(f'{symbol}: RMSE = {np.sqrt(np.mean(error_magnitude[mask]**2)):.2f}, MAE = {np.mean(error_magnitude[mask]):.2f}')
            axis.scatter(dft_magnitude[mask], error_magnitude[mask], s=3, alpha=.3,
                         color=color, rasterized=True, label=symbol)
    axis.text(.03, .97, 'Vector errors (meV/Å)\n' + '\n'.join(force_statistics),
              transform=axis.transAxes, va='top', fontsize=9,
              bbox=dict(facecolor='white', alpha=.85, edgecolor='.8'))
    axis.set(xlabel='DFT force magnitude (meV/Å)',
             ylabel='Force error magnitude ‖MLIP − DFT‖ (meV/Å)',
             title=f'P10 {model_label}: all {len(counts)} clusters (mixed splits)')
    axis.legend(title='Atom type', markerscale=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(destination / 'force_parity.png', dpi=220)
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


def recover_atom_symbols(reference_energy, reference_forces, counts, roles):
    """Recover atom order for older archives, requiring exact reference agreement."""
    import numpy as np
    from ase.io import read
    published = json.loads((OUT / 'p10_published_polar1l_500epochs/published_dataset.json').read_text())
    collection = Path(published['collection'])
    summary = json.loads((collection / 'summary.json').read_text())
    frames, expected_roles = [], []
    for role, filename in [('acquisition', 'finetuning.xyz'), ('evaluation', 'evaluation.xyz')]:
        path = collection / filename
        if sha(path) != summary['files'][role]['sha256']:
            raise ValueError('Dataset hash mismatch while recovering atom symbols')
        subset = read(path, ':')
        frames.extend(subset)
        expected_roles.extend([role] * len(subset))
    if (not np.array_equal(counts, [len(a) for a in frames])
            or not np.array_equal(roles, expected_roles)
            or not np.array_equal(reference_energy, [a.info['REF_energy'] for a in frames])
            or not np.array_equal(reference_forces, np.concatenate([a.arrays['REF_forces'] for a in frames]))):
        raise ValueError('Saved evaluation does not match audited dataset atom order')
    return np.array([symbol for a in frames for symbol in a.get_chemical_symbols()])


def replot(source, destination, selection=None):
    import numpy as np
    source = source.resolve()
    with np.load(source / 'predictions.npz', allow_pickle=False) as saved:
        arrays = [saved[key] for key in ('reference_energy', 'predicted_energy',
                  'reference_forces', 'predicted_forces', 'atom_counts', 'roles')]
        atom_symbols = saved['atom_symbols'] if 'atom_symbols' in saved else None
    metadata = json.loads((source / 'metrics.json').read_text())
    saved_selection = metadata.get('model_selection', 'finet')
    if selection is not None and selection != saved_selection:
        raise ValueError('--model conflicts with saved evaluation model')
    metadata['model_selection'] = saved_selection
    metadata.pop('metrics', None)
    metadata['evaluation_directory'] = str(source)
    if atom_symbols is None:
        atom_symbols = recover_atom_symbols(arrays[0], arrays[2], arrays[4], arrays[5])
    plot_parity(destination, *arrays, metadata, atom_symbols)
    np.savez_compressed(destination / 'predictions.npz',
                        **dict(zip(('reference_energy', 'predicted_energy', 'reference_forces',
                                    'predicted_forces', 'atom_counts', 'roles'), arrays)),
                        atom_symbols=atom_symbols)
    if (source / 'validation_history.csv').exists():
        curves = np.loadtxt(source / 'validation_history.csv', delimiter=',', skiprows=1, ndmin=2)
        plot_history(destination, curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['found', 'finet'], default=None,
                        help='Foundation polar-1-l or fine-tuned epoch 9 (default: finet; saved model when replotting)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--curves-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--from-evaluation', type=Path, metavar='DIRECTORY',
                        help='Regenerate plots from saved evaluation; no inference')
    args = parser.parse_args()
    if args.model == 'found' and args.curves_only:
        parser.error('--curves-only describes fine tuning; use --model finet')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone
    destination = args.output or OUT / 'visualization' / ('p10_' + (args.model or ('saved' if args.from_evaluation else 'finet')) + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    if args.from_evaluation:
        replot(args.from_evaluation, destination, args.model)
        print(destination)
        return
    selection = args.model or 'finet'
    log = ROOT / 'outputsfull/slurm/X_P10_polar1l_500ep_833733.out'
    if selection == 'finet':
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
    model = resolve_model(selection)
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
    atom_symbols = np.array([symbol for a in frames for symbol in a.get_chemical_symbols()])
    np.savez_compressed(destination / 'predictions.npz', reference_energy=re_, predicted_energy=pe,
                        reference_forces=rf, predicted_forces=pf, atom_counts=counts, roles=roles, atom_symbols=atom_symbols)
    plot_parity(destination, re_, pe, rf, pf, counts, roles,
                {'model': str(model), 'model_sha256': sha(model), 'log': str(log),
                 'model_selection': selection,
                 'status': 'unmodified foundation polar-1-l' if selection == 'found' else 'source training timed out; epoch-9 export'}, atom_symbols)
    print(destination)


if __name__ == '__main__':
    main()
