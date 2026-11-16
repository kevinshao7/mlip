#!/usr/bin/env python3
"""Remake all three saved parity plots and energy-outlier GIFs without inference."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from plot_p10_finetuning import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, help='Saved evaluation root with s/m/l folders; default: latest complete parity evaluation')
    args = parser.parse_args()
    source = args.evaluation
    if source is None:
        candidates = sorted((OUT / 'visualization').glob('parity_*'), reverse=True)
        source = next((p for p in candidates if all(
            (p / size / name).is_file() for size in ('s', 'm', 'l')
            for name in ('predictions.npz', 'metrics.json'))), None)
        if source is None:
            parser.error('No complete saved s/m/l parity evaluation found')
    for size in ('s', 'm', 'l'):
        for name in ('predictions.npz', 'metrics.json'):
            if not (source / size / name).is_file():
                parser.error(f'Missing saved evaluation: {source / size / name}')
    output = OUT / 'visualization' / ('energy_outliers_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    scripts = Path(__file__).resolve().parent
    print(f'Saved evaluations: {source}\nNew figures and GIFs: {output}', flush=True)
    for size in ('s', 'm', 'l'):
        subprocess.run([sys.executable, str(scripts / 'plot_parity.py'),
                        '--from-evaluation', str(source / size), '--output', str(output / size)], check=True)
        subprocess.run([sys.executable, str(scripts / 'visualize_saved_outliers.py'), str(source / size),
                        '--output', str(output / size / 'outlier_gifs')], check=True)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()
