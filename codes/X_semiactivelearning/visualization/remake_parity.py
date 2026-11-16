#!/usr/bin/env python3
"""Remake all three saved parity plots and energy-outlier GIFs without inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from plot_p10_finetuning import OUT, replot, sha


def restore_gifs(source, size):
    """Reuse completed matching GIFs from the former separate output layout."""
    destination = source / size / 'outlier_gifs'
    if destination.exists():
        return
    for old in sorted((OUT / 'visualization').glob('energy_outliers_*'), reverse=True):
        record = old / size / 'outlier_gifs/outliers.json'
        if not record.is_file():
            continue
        metadata = json.loads(record.read_text())
        if (Path(metadata['evaluation_directory']).resolve() != (source / size).resolve()
                or metadata['predictions_sha256'] != sha(source / size / 'predictions.npz')):
            continue
        if len(list(record.parent.glob('*.gif'))) == 5:
            shutil.copytree(record.parent, destination)
            print(f'Reused completed GIFs: {destination}', flush=True)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, help='Saved evaluation root with s/m/l folders; default: latest complete parity evaluation')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
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
    scripts = Path(__file__).resolve().parent
    print(f'Plots, predictions and GIFs together: {source}', flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for size in ('s', 'm', 'l'):
        folder = source / size
        # Preserve prior renderings inside this evaluation, then redraw saved data.
        history = folder / 'plot_history' / stamp
        history.mkdir(parents=True, exist_ok=False)
        for name in ('energy_parity.png', 'force_parity.png', 'metrics.json', 'energy_outliers.json'):
            if (folder / name).is_file():
                shutil.copy2(folder / name, history / name)
        replot(folder, folder)
        restore_gifs(source, size)
        if not (folder / 'outlier_gifs').exists():
            subprocess.run([sys.executable, str(scripts / 'visualize_saved_outliers.py'), str(folder),
                            '--output', str(folder / 'outlier_gifs')], check=True)
    print(f'Finished: {source}', flush=True)


if __name__ == '__main__':
    main()
