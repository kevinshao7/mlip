"""Package audited acquisition/evaluation geometries in one extended XYZ file."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from extract_partial_clusters import verify_workspace
import p10_batch as batch


def package(workspace):
    workspace = workspace.resolve()
    verify_workspace(workspace)
    audit_path = workspace / 'cluster_quality.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Geometry audit has not passed')
    summary = workspace / 'cycles/P10p0000GPa/analysis/summary.json'
    if audit['analysis_sha256'] != batch.sha(summary):
        raise ValueError('Analysis changed after geometry audit')
    if audit['snapshot_manifest_sha256'] != batch.sha(workspace / 'snapshot_manifest.json'):
        raise ValueError('Snapshot manifest changed after audit')
    cfg = json.loads((workspace / 'campaign_config.json').read_text())['clusters']
    sources = {role: workspace / 'cycles/P10p0000GPa/clusters' / role / 'clusters.extxyz'
               for role in ['acquisition', 'evaluation']}
    expected = {role: cfg[f'{role}_count'] for role in sources}
    total = sum(expected.values())
    if audit['total'] != total:
        raise ValueError('Audit/config count mismatch')
    for role, path in sources.items():
        if batch.sha(path) != audit['input_sha256'][role]:
            raise ValueError(f'Cluster file changed after audit: {role}')
    output = workspace / f'clusters_{total}.xyz'
    manifest_path = workspace / f'clusters_{total}_manifest.json'
    if output.exists() or manifest_path.exists():
        raise FileExistsError('Preserve the existing packaged dataset')
    partial = output.with_suffix('.xyz.partial')
    with partial.open('xb') as dst:
        for path in sources.values():
            with path.open('rb') as src:
                shutil.copyfileobj(src, dst)
    from ase.io import iread
    counts = Counter()
    keys = set()
    records = []
    for index, atoms in enumerate(iread(partial, format='extxyz')):
        counts[atoms.info['cluster_kind']] += 1
        key = (atoms.info['source_trajectory_id'], int(atoms.info['source_frame']),
               int(atoms.info['central_allowed_component']))
        if key in keys:
            raise ValueError(f'Duplicate source in combined file: {key}')
        keys.add(key)
        records.append({'index': index, 'cluster_kind': atoms.info['cluster_kind'],
                        'source_trajectory_id': key[0], 'source_frame': key[1],
                        'central_allowed_component': key[2], 'atoms': len(atoms),
                        'charge': int(atoms.info['charge']),
                        'multiplicity': int(atoms.info['multiplicity'])})
    if dict(counts) != expected:
        raise ValueError(f'Combined role counts mismatch: {dict(counts)} != {expected}')
    partial.rename(output)
    batch.write_json(manifest_path, {
        'format': 'extended XYZ; unlabelled isolated cluster geometries',
        'file': output.name, 'sha256': batch.sha(output), 'total': total,
        'counts': dict(counts), 'order': ['acquisition', 'evaluation'],
        'records': records,
        'indexing': 'zero based; acquisition first, evaluation follows',
        'geometry_audit_sha256': batch.sha(audit_path),
        'snapshot_manifest_sha256': batch.sha(workspace / 'snapshot_manifest.json'),
        'sources': {role: {'file': str(p.relative_to(workspace)), 'sha256': batch.sha(p)}
                    for role, p in sources.items()},
        'training_rule': 'Never train on cluster_kind=evaluation; retain original temporal validation partitions',
        'correlation_warning': 'Several centres can share one frame; adjacent frames are correlated',
        'dft_status': 'No DFT labels calculated by extraction or packaging'})
    print(output)
    print(manifest_path)
    return output


def git_package(workspace):
    """Copy a compact audited dataset; keep live MD and restart paths intact."""
    workspace = workspace.resolve()
    original = json.loads((workspace / 'clusters_2000_manifest.json').read_text())
    xyz = workspace / original['file']
    audit = workspace / 'cluster_quality.json'
    if original['total'] != 2000 or original['counts'] != {'acquisition': 1950, 'evaluation': 50}:
        raise ValueError('Expected the audited 2000-cluster package')
    if batch.sha(xyz) != original['sha256'] or batch.sha(audit) != original['geometry_audit_sha256']:
        raise ValueError('Dataset or geometry audit changed')
    plan = json.loads((workspace / 'source_plan.json').read_text())
    pressure = plan['pressure_gpa']
    stage = batch.MLIP / 'outputsfull/X_semiactivelearning' / f'P{pressure:g}GPa'
    stamp = workspace.name.removeprefix('partial_2000_')
    release = stage / 'intermediate_cut_clusters' / f'polar1l_2000_{stamp}'
    if release.exists():
        checks = json.loads((release / 'checksums.json').read_text())
        for name, digest in checks.items():
            if batch.sha(release / name) != digest:
                raise ValueError(f'Existing Git package changed: {name}')
        print(release)
        return release
    provenance = release / 'provenance'
    provenance.mkdir(parents=True)
    shutil.copyfile(xyz, release / 'clusters_2000.xyz')
    files = {
        'original_package_manifest.json': workspace / 'clusters_2000_manifest.json',
        'cluster_quality.json': audit,
        'snapshot_manifest.json': workspace / 'snapshot_manifest.json',
        'source_plan.json': workspace / 'source_plan.json',
        'extraction_config.json': workspace / 'campaign_config.json',
        'analysis_summary.json': workspace / 'cycles/P10p0000GPa/analysis/summary.json',
        'thermo_review.json': workspace / 'cycles/P10p0000GPa/analysis/thermo_review.json',
        'approval_acquisition.json': workspace / 'cycles/P10p0000GPa/approval_acquisition.json',
    }
    for role in ['acquisition', 'evaluation']:
        files[f'{role}_selection.csv'] = workspace / 'cycles/P10p0000GPa/clusters' / role / 'selection.csv'
    for name, source in files.items():
        shutil.copyfile(source, provenance / name)
    portable = {key: value for key, value in original.items() if key != 'sources'}
    portable.update(geometry_audit='provenance/cluster_quality.json',
                    snapshot_manifest='provenance/snapshot_manifest.json',
                    original_package_manifest='provenance/original_package_manifest.json',
                    raw_data_policy='Historical source paths are provenance; raw MD/checkpoints are external to this Git package')
    batch.write_json(release / 'clusters_2000_manifest.json', portable)
    (release / 'README.md').write_text(
        '# P10GPa intermediate cut clusters: polar-1-l, 2000 geometries\n\n'
        '`clusters_2000.xyz` is extended XYZ: 1950 acquisition records first, then 50 evaluation records.\n'
        'Keep `cluster_kind=evaluation` out of training. Source frame/replica, charge and multiplicity\n'
        'are retained in every record and in `clusters_2000_manifest.json`. No DFT labels are present.\n\n'
        'Source: ten independent GPU MD runs, 10 GPa, 1920 K, 0.5 fs, periodic MTK NPT;\n'
        '91–92.5 ps partial production, cut from 10.5 ps onward. Each isolated cluster has\n'
        'six whole molecules. Nearby frames/centres are correlated; no equilibrium claim.\n\n'
        '`provenance/` contains immutable historical records; absolute source paths there\n'
        'identify the original machine and are not required to read this portable dataset.\n'
        '`checksums.json` covers every file in this package except itself. Raw trajectories,\n'
        'model weights and restarts are deliberately outside the Git package.\n')
    batch.write_json(release / 'checksums.json', {
        str(p.relative_to(release)): batch.sha(p) for p in sorted(release.rglob('*')) if p.is_file()})
    raw_md = batch.MLIP / plan['output_relative'] / 'cycles/P10p0000GPa/md'
    for name in ['gpu_md_results', 'cpu_dft_results']:
        (stage / name).mkdir(exist_ok=True)
    link = stage / 'gpu_md_results' / plan['run_id']
    if not link.exists() and not link.is_symlink():
        import os
        link.symlink_to(os.path.relpath(raw_md, link.parent), target_is_directory=True)
    batch.write_json(stage / 'gpu_md_results/index.json', {
        'pressure_gpa': pressure, 'stage': 'gpu_md_results', 'model': 'polar-1-l',
        'source_plan': plan['plan_relative'], 'replicas': 10,
        'raw_md_directory': str(raw_md.relative_to(batch.MLIP)),
        'storage': 'local raw data; original paths retained for exact restart'})
    batch.write_json(stage / 'cpu_dft_results/index.json', {
        'pressure_gpa': pressure, 'stage': 'cpu_dft_results',
        'status': 'not_run_for_this_cluster_dataset',
        'input_dataset': str(release.relative_to(batch.MLIP)),
        'prerequisite': 'target-site method pilot and explicit DFT launch authorization'})
    for name, text in {
        'README.md': '# X data at 10 GPa\n\n`gpu_md_results/`: local GPU MD index and links.\n'
                     '`intermediate_cut_clusters/`: compact Git-trackable extracted geometries.\n'
                     '`cpu_dft_results/`: CPU DFT stage index; no calculations run for this dataset.\n'
                     'Original raw paths and restart dependencies remain intact.\n',
        'gpu_md_results/README.md': '# P10GPa GPU MD results\n\nRead `index.json`. Local run links lead to the ten original replica directories.\n'
                                    'Raw trajectories/restarts are not included in Git. Do not move them.\n',
        'cpu_dft_results/README.md': '# P10GPa CPU DFT results\n\nRead `index.json`. No DFT results exist for this cluster release yet.\n'
                                     'Keep future site/run-specific ORCA results here, separate from unlabelled cut clusters.\n',
        'intermediate_cut_clusters/README.md': '# P10GPa intermediate cut clusters\n\nEach dated release contains a portable XYZ, manifest, provenance and checksums.\n'
                                              'Read the workflow partial_clusters_active.json pointer; do not select by date.\n',
    }.items():
        (stage / name).write_text(text)
    pointer_path = batch.ROOT / 'partial_clusters_active.json'
    pointer = json.loads(pointer_path.read_text())
    pointer.update(file=str((release / 'clusters_2000.xyz').relative_to(batch.MLIP)),
                   manifest=str((release / 'clusters_2000_manifest.json').relative_to(batch.MLIP)),
                   geometry_audit=str((provenance / 'cluster_quality.json').relative_to(batch.MLIP)),
                   dataset_directory=str(release.relative_to(batch.MLIP)),
                   data_stage='P10GPa/intermediate_cut_clusters')
    batch.write_json(pointer_path, pointer)
    print(release)
    return release


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--git-package', action='store_true', help='Export the compact audited dataset into pressure/stage-labelled folders')
    args = parser.parse_args()
    (git_package if args.git_package else package)(args.workspace)
