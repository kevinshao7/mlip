"""Plot logged validation loss and RMSEs without model inference."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR', '/tmp/polar-history-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stage-two', type=int, required=True)
    args = parser.parse_args()
    raw = args.log.read_text()
    if 'Training complete' not in raw or not raw.rstrip().endswith('Done'):
        raise ValueError('Expected a completed training and final evaluation log')
    model_match = re.match(r'polar1([sml])_', args.log.name)
    if not model_match:
        raise ValueError('Expected a POLAR-1-S/M/L run log filename')
    model = 'POLAR-1-' + model_match[1].upper()
    completion = re.search(r'(\d{4}-\d{2}-\d{2})[^\n]*INFO: Done\s*$', raw)
    if not completion:
        raise ValueError('Missing dated completion record')
    pattern = r'Epoch (\d+): head: Default, loss=([\d.eE+-]+), MAE_E_per_atom=\s*([\d.eE+-]+) meV, MAE_F=\s*([\d.eE+-]+) meV / A'
    rows = [(int(e), float(l), float(en), float(f)) for e, l, en, f in re.findall(pattern, raw)]
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate epoch metrics')
    results = args.log.parents[1] / 'results' / (args.log.stem + '_train.txt')
    evaluations = {}
    for line in results.read_text().splitlines():
        record = json.loads(line)
        if record.get('mode') == 'eval' and record.get('epoch') is not None and record.get('head') == 'Default':
            epoch = record['epoch']
            if epoch in evaluations:
                raise ValueError(f'Duplicate validation epoch {epoch}')
            evaluations[epoch] = record
    rmse_rows = []
    for epoch, loss, energy, force in rows:
        record = evaluations[epoch]
        if abs(record['loss'] - loss) > 1e-7:
            raise ValueError(f'Log/results mismatch at epoch {epoch}')
        values = (record['loss'], 1000 * record['rmse_e_per_atom'], 1000 * record['rmse_f'])
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f'Invalid metric at epoch {epoch}')
        rmse_rows.append((epoch, *values))
    rows = rmse_rows
    best_energy = min(rows, key=lambda row: row[2])
    best_force = min(rows, key=lambda row: row[3])
    annotation = (f'Best validation energy RMSE: {best_energy[2]:.2f} meV/atom (epoch {best_energy[0]})\n'
                  f'Best validation force RMSE: {best_force[3]:.2f} meV/Å (epoch {best_force[0]})')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'validation_history.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['epoch', 'validation_loss', 'energy_RMSE_meV_per_atom', 'force_RMSE_meV_per_A'])
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    specifications = [('loss', 'Validation weighted loss', 'Weighted loss (log scale)', 1),
                      ('energy_error', 'Validation energy error', 'Energy RMSE (meV/atom)', 2),
                      ('force_error', 'Validation force error', 'Force-component RMSE (meV/Å)', 3)]
    for name, title, ylabel, column in specifications:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for stage, color, select in [('Stage one', '#2563a6', lambda e: e < args.stage_two),
                                     ('Stage two', '#d05d20', lambda e: e >= args.stage_two)]:
            subset = [r for r in rows if select(r[0])]
            ax.plot([r[0] for r in subset], [r[column] for r in subset], color=color, label=stage, linewidth=1.8)
        ax.axvline(args.stage_two, color='gray', linestyle='--', linewidth=1, label=f'Stage two starts: {args.stage_two}')
        ax.set(xlabel='Epoch (logged every 4 epochs)', ylabel=ylabel,
               title=f'{model} · {title}\nCompleted {completion[1]} · seed 123')
        if column == 1:
            ax.set_yscale('log')
        else:
            best = best_energy if column == 2 else best_force
            ax.scatter(best[0], best[column], marker='*', s=130, color='#187344', zorder=5)
        note = annotation
        if column == 1:
            note += '\nLoss weights change at stage two; cross-stage loss is not comparable.'
        fig.text(.5, 1.01, note, ha='center', va='bottom', fontsize=10)
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        fig.savefig(args.output / f'{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    (args.output / 'provenance.json').write_text(json.dumps({
        'source_log': str(args.log.resolve()), 'source_log_sha256': hashlib.sha256(args.log.read_bytes()).hexdigest(),
        'source_results': str(results.resolve()), 'source_results_sha256': hashlib.sha256(results.read_bytes()).hexdigest(),
        'metric_scope': 'Logged validation history; not held-out evaluation',
        'stage_two_epoch': args.stage_two, 'logged_epochs': len(rows),
        'best_energy_RMSE': {'epoch': best_energy[0], 'meV_per_atom': best_energy[2]},
        'best_force_component_RMSE': {'epoch': best_force[0], 'meV_per_A': best_force[3]},
        'last_logged_metrics': dict(zip(['epoch', 'loss', 'energy_RMSE_meV_per_atom', 'force_RMSE_meV_per_A'], rows[-1]))
    }, indent=2) + '\n')
    print(f'{len(rows)} validation records; final logged metrics: {rows[-1]}')


if __name__ == '__main__':
    main()
