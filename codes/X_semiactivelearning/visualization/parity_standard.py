"""Single energy/force residual plot standard; inputs use eV and eV/Angstrom."""
import json


def plot_parity(destination, re_, pe, rf, pf, counts, roles, metadata, atom_symbols):
    import numpy as np
    import matplotlib.pyplot as plt
    model_label = metadata.get('label') or ('Foundation polar-1-l' if metadata.get('model_selection') == 'found' else 'Fine-tuned epoch 9')
    re_, pe, rf, pf, counts, roles = map(np.asarray, (re_, pe, rf, pf, counts, roles))
    if (counts.ndim != 1 or len(counts) == 0 or counts.dtype.kind not in 'iu'
            or (counts <= 0).any() or re_.shape != counts.shape or pe.shape != counts.shape
            or roles.shape != counts.shape or rf.shape != (int(counts.sum()), 3)
            or pf.shape != rf.shape or not all(np.isfinite(a).all() for a in (re_, pe, rf, pf))):
        raise ValueError('Invalid prediction array shapes, counts or finite values')
    metrics = {}
    for name, cluster_mask in [('all', np.ones(len(counts), dtype=bool))] + [(str(role), roles == role) for role in np.unique(roles)]:
        if not cluster_mask.any():
            continue
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
    for mask, label in [(~held_out, metadata.get('dataset_label', 'Acquisition (includes train/validation)')), (held_out, 'Held-out evaluation')]:
        axis.scatter(1000 * re_[mask] / counts[mask], 1000 * (pe[mask] - re_[mask]) / counts[mask],
                     s=3, alpha=.3, rasterized=True, label=label)
    energy_errors = 1000 * (pe - re_) / counts
    worst = np.argsort(-np.abs(energy_errors), kind='stable')[:5]
    axis.scatter(1000 * re_[worst] / counts[worst], energy_errors[worst],
                 s=55, facecolors='none', edgecolors='black', label='Five largest |ΔE|/atom')
    for rank, index in enumerate(worst, 1):
        axis.annotate(f'{rank}: #{index}', (1000 * re_[index] / counts[index], energy_errors[index]),
                      xytext=(7, 7 if rank % 2 else -14), textcoords='offset points', fontsize=8)
    axis.text(.03, .97,
              f'All clusters: RMSE = {np.sqrt(np.mean(energy_errors**2)):.2f} meV/atom\n'
              f'MAE = {np.mean(np.abs(energy_errors)):.2f} meV/atom',
              transform=axis.transAxes, va='top', fontsize=9,
              bbox=dict(facecolor='white', alpha=.85, edgecolor='.8'))
    axis.axhline(0, color='k', ls='--', lw=1)
    axis.set(xlabel='DFT absolute electronic energy (meV/atom)',
             ylabel='MLIP − DFT energy (meV/atom)',
             title=f'P10 {model_label}: all {len(counts)} clusters')
    axis.legend(markerscale=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(destination / 'energy_parity.png', dpi=220)
    plt.close(fig)
    (destination / 'energy_outliers.json').write_text(json.dumps([
        {'rank': rank, 'cluster_index': int(index), 'index_convention': 'zero-based saved prediction order',
         'atom_count': int(counts[index]), 'role': str(roles[index]),
         'energy_error_meV_per_atom': float(energy_errors[index])}
        for rank, index in enumerate(worst, 1)], indent=2) + '\n')
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
             title=f'P10 {model_label}: all {len(counts)} clusters')
    axis.legend(title='Atom type', markerscale=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(destination / 'force_parity.png', dpi=220)
    plt.close(fig)
    (destination / 'metrics.json').write_text(json.dumps({**metadata, 'metrics': metrics,
        'note': 'All-data parity is descriptive; correlated acquisition frames are not independent test samples.'}, indent=2) + '\n')
