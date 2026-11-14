"""Plot logged validation loss and MAEs without model inference."""
import argparse
import csv
import hashlib
import json
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
    pattern = r'Epoch (\d+): head: Default, loss=([\d.eE+-]+), MAE_E_per_atom=\s*([\d.eE+-]+) meV, MAE_F=\s*([\d.eE+-]+) meV / A'
    rows = [(int(e), float(l), float(en), float(f)) for e, l, en, f in re.findall(pattern, raw)]
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate epoch metrics')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'validation_history.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['epoch', 'validation_loss', 'energy_MAE_meV_per_atom', 'force_MAE_meV_per_A'])
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    specifications = [('loss', 'Validation weighted loss', 'Weighted loss (log scale)', 1),
                      ('energy_error', 'Validation energy error', 'Energy MAE (meV/atom)', 2),
                      ('force_error', 'Validation force error', 'Force-component MAE (meV/Å)', 3)]
    for name, title, ylabel, column in specifications:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for stage, color, select in [('Stage one', '#2563a6', lambda e: e < args.stage_two),
                                     ('Stage two', '#d05d20', lambda e: e >= args.stage_two)]:
            subset = [r for r in rows if select(r[0])]
            ax.plot([r[0] for r in subset], [r[column] for r in subset], color=color, label=stage, linewidth=1.8)
        ax.axvline(args.stage_two, color='gray', linestyle='--', linewidth=1, label=f'Stage two starts: {args.stage_two}')
        ax.set(xlabel='Epoch (logged every 4 epochs)', ylabel=ylabel,
               title=f'POLAR-1-L · {title}\nCompleted 14 November 2026 · seed 123')
        if column == 1:
            ax.set_yscale('log')
            ax.text(.98, .97, 'Loss weights change at stage two;\nloss values across stages are not comparable.',
                    transform=ax.transAxes, va='top', ha='right', fontsize=9)
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        fig.savefig(args.output / f'{name}.png', dpi=180)
        plt.close(fig)
    (args.output / 'provenance.json').write_text(json.dumps({
        'source_log': str(args.log.resolve()), 'source_log_sha256': hashlib.sha256(args.log.read_bytes()).hexdigest(),
        'metric_scope': 'Logged validation history; not held-out evaluation',
        'stage_two_epoch': args.stage_two, 'logged_epochs': len(rows),
        'last_logged_metrics': dict(zip(['epoch', 'loss', 'energy_MAE_meV_per_atom', 'force_MAE_meV_per_A'], rows[-1]))
    }, indent=2) + '\n')
    print(f'{len(rows)} validation records; final logged metrics: {rows[-1]}')


if __name__ == '__main__':
    main()
