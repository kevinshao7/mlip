"""Snapshot checkpoint-committed P10 data, review, then reuse X cluster extraction.

This tool never launches MD/DFT/training or changes the active P10 launch plan.
Run prepare first, review diagnostics, then explicitly approve and extract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'production'))
import campaign
import p10_batch as batch
import numpy as np
from workflow_species import iter_xyz_frames


def copy_prefix(source, target, size):
    """Exclude uncommitted writes; fail rather than accept a truncated prefix."""
    target.parent.mkdir(parents=True, exist_ok=True)
    remaining = size
    with source.open('rb') as src, target.open('xb') as dst:
        while remaining:
            block = src.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f'{source}: shorter than committed offset {size}')
            dst.write(block)
            remaining -= len(block)


def bind_workspace(workspace):
    workspace = workspace.resolve()
    campaign.OUT = workspace
    campaign.RUN_ID = workspace.name
    return workspace


def verify_workspace(workspace):
    manifest = json.loads((workspace / 'snapshot_manifest.json').read_text())
    for relative, digest in manifest['sha256'].items():
        if batch.sha(workspace / relative) != digest:
            raise ValueError(f'Snapshot changed: {relative}')
    for relative, digest in manifest['source_sha256'].items():
        if batch.sha(batch.MLIP / relative) != digest:
            raise ValueError(f'Extraction code changed; use a fresh workspace: {relative}')
    return manifest


def prepare(args):
    plan = batch.load_plan(args.plan)
    model, binding = batch.binding(plan)
    source_out = batch.MLIP / plan['output_relative']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.total < 1 or not 0 < args.evaluation < args.total:
        raise ValueError('Require 0 < evaluation count < total count')
    workspace = source_out / 'cluster_extractions' / f'partial_{args.total}_{stamp}'
    workspace.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((source_out / 'campaign_config.json').read_text())
    cfg['campaign_name'] = workspace.name
    cfg['clusters']['acquisition_count'] = args.total - args.evaluation
    cfg['clusters']['evaluation_count'] = args.evaluation
    batch.write_json(workspace / 'campaign_config.json', cfg)
    batch.write_json(workspace / 'source_plan.json', plan)
    records = []
    for replica in plan['replicas']:
        source = source_out / 'cycles/P10p0000GPa/md' / replica['id']
        target = workspace / 'snapshots' / replica['id']
        target.mkdir(parents=True)
        # Copy the atomic restart before reading its offsets, never trust EOF or
        # the independently updated status JSON as a transaction boundary.
        shutil.copyfile(source / 'restart.npz', target / 'restart.npz')
        with np.load(target / 'restart.npz', allow_pickle=False) as checkpoint:
            meta = json.loads(str(checkpoint['metadata']))
        expected = {'model_sha256': binding['sha256'],
                    'batch_config_sha256': plan['config_sha256'],
                    'pressure_gpa': plan['pressure_gpa'],
                    'temperature_K': plan['temperature_K'],
                    'timestep_fs': plan['md']['timestep_fs'], 'pbc': [True] * 3}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{source}: checkpoint model/config/physical settings mismatch')
        paths = list(source.glob('production_*.xyz'))
        thermo = list(source.glob('*_thermo.txt'))
        if len(paths) != 1 or len(thermo) != 1:
            raise ValueError(f'{source}: expected exactly one trajectory and thermo file')
        copy_prefix(paths[0], target / paths[0].name, meta['trajectory_bytes'])
        copy_prefix(thermo[0], target / thermo[0].name, meta['thermo_bytes'])
        for name in ['manifest.json', 'model_runtime.json']:
            shutil.copyfile(source / name, target / name)
        run = json.loads((target / 'manifest.json').read_text())
        if run['replicas'][0]['seed'] != replica['seed']:
            raise ValueError(f'{source}: seed mismatch')
        frames = 0
        for _, frame in iter_xyz_frames(target / paths[0].name):
            if Counter(frame.symbols) != Counter(H=250, N=50, O=50):
                raise ValueError(f'{source}: unexpected composition')
            if not np.isfinite(frame.positions).all() or not np.linalg.det(frame.cell) > 0:
                raise ValueError(f'{source}: invalid frame geometry')
            frames += 1
        production_steps = max(0, meta['step'] - meta['pressure_ramp_steps'] - meta['temperature_ramp_steps'])
        if frames != production_steps // meta['save_interval_steps'] or not frames:
            raise ValueError(f'{source}: committed frame count disagrees with checkpoint')
        data = np.loadtxt(target / thermo[0].name, ndmin=2)
        if len(data) != frames or not np.isfinite(data).all():
            raise ValueError(f'{source}: thermo/frame mismatch')
        expected_times = (meta['pressure_ramp_steps'] + meta['temperature_ramp_steps']
                          + np.arange(1, frames + 1) * meta['save_interval_steps']) * meta['timestep_fs']
        if not np.allclose(data[:, 0], expected_times, rtol=0, atol=1e-8):
            raise ValueError(f'{source}: saved-time sequence mismatch')
        records.append({'id': replica['id'], 'seed': replica['seed'], 'source': str(paths[0]),
                        'snapshot': str((target / paths[0].name).relative_to(workspace)),
                        'thermo': str((target / thermo[0].name).relative_to(workspace)),
                        'frames': frames, 'checkpoint': meta,
                        'production_time_ps': production_steps * meta['timestep_fs'] / 1000,
                        'complete': meta['step'] == meta['total_steps']})
    sources = [Path(__file__), *sorted((ROOT / 'production').glob('*.py')),
               ROOT / 'autonomous_policy.json']
    manifest = {'created_utc': campaign.now(), 'source_plan': plan['plan_relative'],
                'model_sha256': binding['sha256'], 'model': str(model), 'replicas': records,
                'completion_policy': 'checkpoint-committed partial production; no completion claim',
                'sha256': {str(p.relative_to(workspace)): batch.sha(p)
                           for p in workspace.rglob('*') if p.is_file()},
                'source_sha256': {str(p.relative_to(batch.MLIP)): batch.sha(p) for p in sources}}
    batch.write_json(workspace / 'snapshot_manifest.json', manifest)
    batch.notebook(f'Partial-run snapshots: {workspace}; frame counts {[r["frames"] for r in records]}.',
                   'All ten checkpoints verified; incomplete runs are not 1 ns completion evidence.',
                   'Review pressure including kinetic contribution, volume and species before selecting a window.',
                   'Immutable committed prefixes prevent live writes and restart rollbacks changing the extraction.',
                   f'Extraction budget {args.total}; MD settings/model retained. No jobs submitted.')
    print(workspace, flush=True)


def analyze(args):
    workspace = bind_workspace(args.workspace)
    manifest = verify_workspace(workspace)
    argv = ['analyze', '--pressure', '10', '--workers', str(args.workers),
            '--production-start-frame', str(args.start_frame), '--frame-time-ps', '0.5']
    for replica in manifest['replicas']:
        if replica['frames'] <= args.start_frame:
            raise ValueError('Review window leaves a replica empty')
        argv += ['--trajectory', str(workspace / replica['snapshot'])]
    parsed = campaign.parser().parse_args(argv)
    parsed.func(parsed)
    # Existing extxyz diagnostic uses virial stress without the kinetic term.
    # Review the actual NPT pressure from the runner's include_ideal_gas=True log.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    review = []
    for replica in manifest['replicas']:
        data = np.loadtxt(workspace / replica['thermo'], ndmin=2)[args.start_frame:]
        time = (data[:, 0] - 4000) / 1000
        axes[0].plot(time, data[:, 2], label=replica['id'], linewidth=.7)
        axes[1].plot(time, data[:, 3], label=replica['id'], linewidth=.7)
        blocks = []
        for block in np.array_split(data, min(4, len(data))):
            blocks.append({'frames': len(block), 'temperature_K': float(block[:, 1].mean()),
                           'pressure_GPa': float(block[:, 2].mean()),
                           'volume_A3': float(block[:, 3].mean())})
        review.append({'id': replica['id'], 'frames': len(data),
                       'first_production_time_ps': float(time[0]), 'last_production_time_ps': float(time[-1]),
                       'blocks': blocks, 'pressure_GPa': float(data[:, 2].mean()),
                       'volume_A3': float(data[:, 3].mean())})
    axes[0].axhline(10, color='black', linestyle='--')
    axes[0].set_ylabel('pressure incl. kinetic / GPa')
    axes[0].legend(ncol=5, fontsize='small')
    axes[1].set(ylabel='volume / Angstrom³', xlabel='original production time / ps')
    fig.tight_layout()
    analysis = workspace / 'cycles/P10p0000GPa/analysis'
    fig.savefig(analysis / 'thermo_pressure_volume.png', dpi=150)
    plt.close(fig)
    batch.write_json(analysis / 'thermo_review.json', {
        'pressure_source': 'checkpoint-committed thermo, include_ideal_gas=True',
        'time_source': 'thermo time_fs minus original 4 ps ramps; no window time reset',
        'uncertainty_policy': 'four chronological block means; not IID standard errors or equilibrium proof',
        'replicas': review})
    print(json.dumps(review, indent=2), flush=True)


def extract(args):
    workspace = bind_workspace(args.workspace)
    verify_workspace(workspace)
    analysis = workspace / 'cycles/P10p0000GPa/analysis'
    summary = json.loads((analysis / 'summary.json').read_text())
    if not summary['cluster_eligible'] or len(summary['trajectories']) != 10:
        raise ValueError('Ten-replica species eligibility failed')
    if not args.reason.strip():
        raise ValueError('Record explicit review authorization and its scientific limitations')
    for role in ['evaluation', 'acquisition']:
        if (workspace / 'cycles/P10p0000GPa/clusters' / role).exists():
            raise FileExistsError('Extraction exists; preserve it and prepare a new workspace')
    approval = campaign.parser().parse_args(['approve', '--pressure', '10', '--reason', args.reason])
    approval.func(approval)
    cfg = campaign.config()['clusters']
    for role, count in [('evaluation', cfg['evaluation_count']), ('acquisition', cfg['acquisition_count'])]:
        parsed = campaign.parser().parse_args(['clusters', '--pressure', '10', '--kind', role,
                                              '--count', str(count), '--workers', str(args.workers),
                                              '--visualization-limit', '2'])
        parsed.func(parsed)
    from cluster_quality import audit_sets
    paths = {role: workspace / 'cycles/P10p0000GPa/clusters' / role / 'clusters.extxyz'
             for role in ['acquisition', 'evaluation']}
    audit = audit_sets(paths, summary)
    total = cfg['evaluation_count'] + cfg['acquisition_count']
    if audit['total'] != total:
        raise ValueError(f'Expected exactly {total} clusters')
    audit['analysis_sha256'] = batch.sha(analysis / 'summary.json')
    audit['snapshot_manifest_sha256'] = batch.sha(workspace / 'snapshot_manifest.json')
    batch.write_json(workspace / 'cluster_quality.json', audit)
    batch.notebook(f'Extracted and audited {cfg["acquisition_count"]} acquisition +{cfg["evaluation_count"]} evaluation clusters: {workspace}; audit={workspace / "cluster_quality.json"}.',
                   'Partial trajectories and temporal blocks remain correlated; extraction is not equilibrium or method-pilot certification.',
                   'Inspect cluster galleries and validate the target-site method pilot before separately authorized DFT preparation.',
                   'Separates geometry quality from electronic-method and sampling claims.',
                   f'Extraction count only: {total} requested; no MD/DFT/training launched.')
    print(f'PASS: exactly {total} audited clusters in {workspace}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    sub = commands.add_parser('prepare')
    sub.add_argument('--plan', type=Path)
    sub.add_argument('--total', type=int, default=2000)
    sub.add_argument('--evaluation', type=int, default=50)
    sub.set_defaults(func=prepare)
    for name, func in [('analyze', analyze), ('extract', extract)]:
        sub = commands.add_parser(name)
        sub.add_argument('--workspace', type=Path, required=True)
        sub.add_argument('--workers', type=int, default=4)
        if name == 'analyze':
            sub.add_argument('--start-frame', type=int, default=0)
        else:
            sub.add_argument('--reason', required=True)
        sub.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
