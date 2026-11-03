#!/usr/bin/env python3
"""Sequential MD, cluster, wB97M DFT, training and held-out force-error loop.

--once executes one stage; --watch repeats; --dry-run only prints the next
stage. The previous controller is preserved in deprecated/.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('SAL_RUN_ID', 'wb97m_step5')
import campaign
from workflow_policy import passes_force_gate, policy
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def next_action():
    cfg = policy()
    if not (campaign.OUT / 'model_lineage.json').exists():
        return ['init']
    if not (campaign.OUT / 'method_verified.json').exists():
        pilot = campaign.OUT / 'method_pilot'
        if not (pilot / 'reference.json').exists():
            return ['prepare-method-pilot']
        if not (pilot / 'outputs/pilot.out').exists():
            return ['await-method-pilot']
        return ['verify-method-pilot']
    from workflow_policy import require_method_verified
    require_method_verified()
    pressures = campaign.config()['pressures_gpa']
    expected = list(range(cfg['start_pressure_gpa'], int(pressures[-1])+1, cfg['pressure_increment_gpa']))
    if pressures != expected:
        raise ValueError('Campaign pressure grid disagrees with the 5 GPa policy')
    progress = read(campaign.OUT / 'pressure_progress.json').get('passed', {})
    for pressure in pressures:
        if campaign.tag(pressure) in progress:
            continue
        cdir = campaign.cycle(pressure); prefix = ['--pressure', str(pressure)]; md = cdir / 'md'
        if not (md / 'complete.json').exists():
            if list(md.glob('**/production_*.xyz')):
                raise ValueError(f'Uncertified/incomplete MD at {md}; inspect before resuming; never overwrite')
            return ['md', *prefix, '--run']
        summary = read(cdir / 'analysis/summary.json')
        if not summary:
            return ['analyze', *prefix]
        if summary.get('frames', 0) < campaign.config()['md']['target_production_frames']:
            raise ValueError('MD has fewer complete frames than the configured production target')
        if not summary.get('cluster_eligible'):
            raise ValueError('Species eligibility failed; inspect analysis before acquisition')
        if not (cdir / 'approval_acquisition.json').exists():
            raise ValueError(f'Review {cdir / "analysis/pressure_volume_time_series.png"} and species plot, then record campaign.py approve --pressure {pressure} --reason "equilibration evidence and decision"')
        if not (cdir / 'clusters/evaluation/clusters.extxyz').exists():
            return ['clusters', *prefix, '--kind', 'evaluation']
        rounds = sorted((cdir / 'clusters').glob('acquisition*'), key=lambda p: 1 if p.name == 'acquisition' else int(p.name.split('_r')[-1]))
        name = rounds[-1].name if rounds else 'acquisition'
        if not rounds:
            return ['clusters', *prefix, '--kind', name]
        for kind in ['evaluation', name]:
            ddir = cdir / 'dft' / kind; dft = ['dft', *prefix, '--kind', kind]
            if not (ddir / 'manifest.csv').exists():
                return dft
            with (ddir / 'manifest.csv').open() as stream:
                jobs = list(csv.DictReader(stream))
            if not all((ddir / 'outputs' / f"{row['stem']}.exitcode").exists()
                       or campaign.local_orca_complete(ddir / 'outputs' / f"{row['stem']}.out") for row in jobs):
                return ['await-dft', str(pressure), kind]
            if not (ddir / 'processed.extxyz').exists() or (ddir / 'processed.extxyz').stat().st_size == 0:
                if (ddir / 'collection.csv').exists():
                    raise ValueError(f'No accepted DFT labels in {ddir}; failures preserved, never retried automatically')
                return [*dft, '--collect']
        acquisition = cdir / 'dft' / name / 'processed.extxyz'
        history = read(campaign.OUT / 'model_lineage.json').get('history', [])
        trained = any(entry.get('acquired_sha256', {}).get(str(acquisition.resolve())) == campaign.sha(acquisition) for entry in history)
        if not trained:
            return ['launch-training']
        ddir = cdir / 'dft/evaluation'; report = read(ddir / 'parity_summary.json')
        model = campaign.current_parent()
        model_hash = campaign.sha(Path(model)) if Path(model).is_file() else model
        data_hash = campaign.sha(ddir / 'processed.extxyz')
        if (report.get('dft_functional_basis') != campaign.config()['dft']['functional_basis']
                or report.get('evaluation_scope') != 'reserved_temporal_blocks'
                or report.get('parent_model') != model or report.get('model_sha256') != model_hash
                or report.get('dataset_sha256') != data_hash):
            return ['dft', *prefix, '--kind', 'evaluation', '--parity']
        if report.get('status') != 'complete':
            raise ValueError(f'Evaluation unavailable: {report.get("reason")}')
        if report.get('n_clusters', 0) < cfg['minimum_evaluation_clusters']:
            raise ValueError('Insufficient held-out evaluation labels; advancement blocked')
        if passes_force_gate(report, cfg):
            return ['record-pass', str(pressure)]
        number = 1 if name == 'acquisition' else int(name.split('_r')[-1])
        return ['clusters', *prefix, '--kind', f'acquisition_r{number+1}']
    return ['complete']


def note(text):
    with (ROOT / 'hyperparameters.txt').open('a') as handle:
        handle.write(f'\n{datetime.now(timezone.utc).isoformat()} — {text}\n')


def execute(action):
    if action[0].startswith('await-'):
        print('Submit prepared DFT on Viper or BlueHive, transfer outputs back, then collect; controller awaits terminal job records.', flush=True)
        return
    if action[0] == 'record-pass':
        pressure = float(action[1]); cdir = campaign.cycle(pressure)
        report = read(cdir / 'dft/evaluation/parity_summary.json')
        if not passes_force_gate(report):
            raise ValueError('Force gate no longer passes')
        path = campaign.OUT / 'pressure_progress.json'
        progress = read(path); progress.setdefault('passed', {})[campaign.tag(pressure)] = report
        campaign.write_json(path, progress)
        campaign.event('pressure_force_gate_passed', pressure, [cdir / 'dft/evaluation/parity_summary.json'], [path], **report)
        note(f'Pressure {pressure:g} GPa passed: {report}; next test changes pressure only by +5 GPa, preserving the promoted model and MD recipe.')
        return
    command = ([sys.executable, str(ROOT / 'launch_single_gpu.py')] if action == ['launch-training']
               else [sys.executable, str(ROOT / 'campaign.py'), *action])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = campaign.OUT / 'autonomous_logs' / f'{action[0]}_{stamp}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    note(f'Next stage {action}; evidence: {log}. Uncertainty: stage outcome pending. Next test executes this stage with current settings; no optimizer/model/precision change. Training changes acquisition_dataset only.')
    env = os.environ.copy(); env['XDG_CACHE_HOME'] = str(campaign.MLIP / 'outputsfull/.cache')
    if not os.environ.get('SLURM_JOB_ID'):
        env['CUDA_VISIBLE_DEVICES'] = policy()['md_gpu' if action[0] == 'md' else 'mace_gpu']
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    log.with_suffix('.exitcode').write_text(str(result.returncode)+'\n')
    note(f'Stage {action} exited {result.returncode}; exact evidence: {log}. Next test: inspect output before progression; failed stages stop the controller. Changed factor: none beyond the recorded stage.')
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--once', action='store_true'); modes.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true'); args = parser.parse_args()
    while True:
        action = next_action(); print(action, flush=True)
        if args.dry_run or action == ['complete']:
            return
        execute(action)
        if args.once:
            return
        time.sleep(policy()['poll_interval_seconds'])


if __name__ == '__main__':
    main()
