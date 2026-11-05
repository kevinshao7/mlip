#!/usr/bin/env python3
"""Prepare/check/submit/collect isolated ORCA jobs from the published P10 release."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MLIP = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'production'))
import campaign
import p10_batch

POINTER = ROOT / 'cpu_dft_active.json'
DATASET = ROOT / 'partial_clusters_active.json'
COUNTS = {'acquisition': 1950, 'evaluation': 50}
N_JOBS, PER_JOB, CPUS = 200, 10, 20
METHOD = 'wB97M/def2-TZVPD'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def relative(path):
    return str(Path(path).resolve().relative_to(MLIP))


def dataset_frames(pointer=DATASET):
    """Bind every geometry to the published checksum, audit and source record."""
    from ase.io import read
    import numpy as np
    source = load(pointer)
    xyz, manifest_path, audit_path = (MLIP / source[k] for k in ('file', 'manifest', 'geometry_audit'))
    release = MLIP / source['dataset_directory']
    for name, expected in load(release / 'checksums.json').items():
        if digest(release / name) != expected:
            raise ValueError(f'Released dataset checksum mismatch: {name}')
    manifest, audit = load(manifest_path), load(audit_path)
    if digest(xyz) != source['sha256'] or digest(xyz) != manifest['sha256']:
        raise ValueError('Published XYZ checksum mismatch')
    if digest(audit_path) != manifest['geometry_audit_sha256']:
        raise ValueError('Geometry audit checksum mismatch')
    if audit['status'] != 'passed' or audit['total'] != 2000 or manifest['counts'] != COUNTS:
        raise ValueError('Expected passed audit and 1950 acquisition + 50 evaluation clusters')
    frames = read(xyz, index=':', format='extxyz')
    records = manifest['records']
    if len(frames) != 2000 or len(records) != 2000 or len(audit['clusters']) != 2000:
        raise ValueError('Expected exactly 2000 geometries and audit/manifest records')
    counts, keys, role_frames = Counter(), set(), {role: set() for role in COUNTS}
    role_index = Counter()
    for index, (atoms, record, audited) in enumerate(zip(frames, records, audit['clusters'], strict=True)):
        info = atoms.info
        charge = campaign.formal_charge(atoms.get_chemical_symbols())
        if info['charge'] != charge or record['charge'] != charge or audited['charge'] != charge:
            raise ValueError(f'Formal charge mismatch at cluster {index}')
        if info['multiplicity'] != 1 or info.get('spin', 1) != 1 or record['multiplicity'] != 1:
            raise ValueError(f'Multiplicity mismatch at cluster {index}')
        if (int(sum(atoms.numbers)) - charge) % 2:
            raise ValueError(f'Odd electron count at cluster {index}')
        if atoms.pbc.any() or np.any(atoms.cell[:]) or not np.isfinite(atoms.positions).all():
            raise ValueError(f'Expected finite isolated geometry at cluster {index}')
        role = info['cluster_kind']
        if role not in COUNTS or record['cluster_kind'] != role or audited['role'] != role:
            raise ValueError(f'Role mismatch at cluster {index}')
        if record['index'] != index or audited['index'] != role_index[role]:
            raise ValueError(f'Index mismatch at cluster {index}')
        for field in ('source_trajectory_id', 'source_frame', 'central_allowed_component'):
            if info[field] != record[field]:
                raise ValueError(f'Source provenance mismatch at cluster {index}: {field}')
        if len(atoms) != record['atoms'] or len(atoms) != audited['atoms']:
            raise ValueError(f'Atom count mismatch at cluster {index}')
        if info['molecules'] != 6 or audited['molecules'] != 6 or not info['whole_molecules_preserved'] or not info['allowed_molecules_only']:
            raise ValueError(f'Molecule audit mismatch at cluster {index}')
        if float(info['max_internal_distance_error_angstrom']) > 1e-8:
            raise ValueError(f'Intramolecular placement error at cluster {index}')
        key = tuple(info[k] for k in ('source_trajectory_id', 'source_frame', 'central_allowed_component'))
        if key in keys:
            raise ValueError('Duplicate source frame/centre')
        keys.add(key)
        role_frames[role].add(key[:2])
        counts[role] += 1
        role_index[role] += 1
    if counts != COUNTS or role_frames['acquisition'] & role_frames['evaluation']:
        raise ValueError('Role count or temporal evaluation segregation mismatch')
    return source, frames, records


def shell_script(package, index):
    resources = package['resources']
    bundle = MLIP / package['bundle']
    logs = MLIP / package['output'] / 'slurm'
    python = shlex.quote(str(MLIP / '.venv/bin/python'))
    runner = shlex.quote(str(Path(__file__).resolve()))
    manifest = shlex.quote(str(bundle / 'package.json'))
    return f'''#!/bin/bash -l
#SBATCH --job-name=X_P10_DFT_{index:03d}
#SBATCH --partition={resources['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem={resources['memory_gib']}G
#SBATCH --time={resources['walltime']}
#SBATCH --output={logs}/group_{index:03d}_%j.out
#SBATCH --error={logs}/group_{index:03d}_%j.err
set -euo pipefail
module purge
module load {package['orca_module']}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME={shlex.quote(str(MLIP / 'outputsfull/.cache'))}
cd {shlex.quote(str(MLIP))}
"${{PYTHON_COMMAND:-{python}}}" {runner} --package {manifest} run-group --index {index}
'''


def prepare(args):
    source, frames, records = dataset_frames()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    bundle = ROOT / 'dft/batches' / f'p10_2000_viper_{stamp}'
    output = MLIP / 'outputsfull/X_semiactivelearning/P10GPa/cpu_dft_results' / bundle.name
    bundle.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('slurm', 'submissions'):
        (output / name).mkdir()
    package = {'created_utc': datetime.now(timezone.utc).isoformat(), 'bundle': relative(bundle),
               'output': relative(output), 'dataset': source, 'dataset_pointer_sha256': digest(DATASET),
               'resources': {'partition': args.partition, 'walltime': args.walltime,
                             'memory_gib': args.memory_gib, 'cpus': CPUS},
               'jobs': N_JOBS, 'clusters_per_job': PER_JOB,
               'orca_module': load(ROOT / 'site_profiles.json')['viper']['module'],
               'method': METHOD, 'label_route': 'orca_d4rev_subtraction',
               'energy_reference': 'absolute electronic eV', 'force_units': 'eV/Angstrom',
               'entries': [], 'files': {}, 'source_sha256': {}}
    for index, (atoms, record) in enumerate(zip(frames, records, strict=True)):
        stem = f'cluster_{index:04d}'
        inp = bundle / 'inputs' / f'{stem}.inp'
        inp.parent.mkdir(exist_ok=True)
        inp.write_text(campaign.dft_input(atoms, int(record['charge']), 1, CPUS, METHOD))
        package['entries'].append({**record, 'stem': stem, 'group': index // PER_JOB,
            'role': record['cluster_kind'], 'n_atoms': len(atoms),
            'input_relative': relative(inp), 'input_sha256': digest(inp)})
    for index in range(N_JOBS):
        job = bundle / 'jobs' / f'group_{index:03d}.slurm'
        job.parent.mkdir(exist_ok=True)
        job.write_text(shell_script(package, index))
    submit = bundle / 'submitcpu.sh'
    submit.write_text('#!/bin/bash\nset -euo pipefail\n'
        'bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'exec "${{PYTHON_COMMAND:-{MLIP / ".venv/bin/python"}}}" '
        f'{shlex.quote(str(Path(__file__).resolve()))} --package "$bundle_dir/package.json" submit "$@"\n')
    for path in bundle.rglob('*'):
        if path.is_file():
            package['files'][str(path.relative_to(bundle))] = digest(path)
    for path in [Path(__file__), ROOT / 'production/campaign.py', ROOT / 'production/p10_batch.py',
                 ROOT / 'production/orca_output_validation.py', ROOT / 'production/workflow_policy.py',
                 ROOT.parent / 'C6_wb97m_nodisp/prepare_data.py']:
        package['source_sha256'][relative(path)] = digest(path)
    write_json(bundle / 'package.json', package)
    write_json(POINTER, {'package': relative(bundle / 'package.json'),
                         'sha256': digest(bundle / 'package.json')})
    print(f'Prepared {N_JOBS} jobs x {PER_JOB} clusters; no submission: {bundle}')
    return bundle / 'package.json'


def resolve_package(path=None):
    if path is None:
        pointer = load(POINTER)
        path = MLIP / pointer['package']
        if digest(path) != pointer['sha256']:
            raise ValueError('Active CPU package checksum mismatch')
    return Path(path).resolve()


def check(path, full=True):
    package = load(path)
    bundle = MLIP / package['bundle']
    if (bundle / 'package.json').resolve() != path.resolve():
        raise ValueError('Package location mismatch')
    if package['jobs'] != N_JOBS or package['clusters_per_job'] != PER_JOB or len(package['entries']) != 2000:
        raise ValueError('Expected 200 jobs of 10 clusters')
    if package['resources']['cpus'] != CPUS or package['method'] != METHOD or package['label_route'] != 'orca_d4rev_subtraction':
        raise ValueError('CPU/method/label route mismatch')
    for name, expected in package['source_sha256'].items():
        if digest(MLIP / name) != expected:
            raise ValueError(f'Code changed; prepare a fresh batch: {name}')
    for name, expected in package['files'].items():
        if digest(bundle / name) != expected:
            raise ValueError(f'Prepared file changed: {name}')
    if full:
        if load(DATASET) != package['dataset'] or digest(DATASET) != package['dataset_pointer_sha256']:
            raise ValueError('Active cluster dataset changed; prepare a fresh batch')
        _, frames, records = dataset_frames()
    for index, entry in enumerate(package['entries']):
        if entry['index'] != index or entry['group'] != index // PER_JOB:
            raise ValueError('Group membership mismatch')
        inp = p10_batch.validate_input(entry)
        if '%pal nprocs 20 end' not in inp.read_text():
            raise ValueError('ORCA must use exactly 20 MPI processes')
        if full:
            expected = campaign.dft_input(frames[index], int(records[index]['charge']), 1, CPUS, METHOD)
            if inp.read_text() != expected or entry['role'] != records[index]['cluster_kind']:
                raise ValueError(f'Input geometry/role differs from published cluster {index}')
    if Counter(e['role'] for e in package['entries']) != COUNTS:
        raise ValueError('Dataset split mismatch')
    return package


def submit(path, dry_run=False):
    package = check(path)
    output, bundle = MLIP / package['output'], MLIP / package['bundle']
    if dry_run:
        for index in range(N_JOBS):
            print(f'sbatch --parsable {bundle / "jobs" / f"group_{index:03d}.slurm"}')
        return
    submissions = output / 'submissions'
    submissions.mkdir(parents=True, exist_ok=True)
    # Exclusive claims prevent duplicate submission, including ambiguous failures.
    for index in range(N_JOBS):
        record = submissions / f'group_{index:03d}.json'
        if record.exists():
            previous = load(record)
            if previous['status'] == 'submitted':
                print(f'Already submitted group {index}: {previous["job_id"]}')
                continue
            raise RuntimeError(f'Unresolved submission attempt: {record}; inspect before retrying')
        with record.open('x') as handle:
            json.dump({'status': 'attempting', 'group': index}, handle)
        result = subprocess.run(['sbatch', '--parsable', str(bundle / 'jobs' / f'group_{index:03d}.slurm')],
                                text=True, capture_output=True)
        job_id = result.stdout.strip().split(';')[0]
        state = {'status': 'submitted' if result.returncode == 0 and job_id.isdigit() else 'uncertain_or_failed',
                 'group': index, 'job_id': job_id, 'stdout': result.stdout, 'stderr': result.stderr,
                 'returncode': result.returncode, 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
        write_json(record, state)
        if state['status'] != 'submitted':
            raise RuntimeError(f'Submission stopped; inspect {record}')
        print(f'Group {index:03d}: job {job_id}', flush=True)


def run_group(path, index):
    if not 0 <= index < N_JOBS:
        raise ValueError('Group index must be 0..199')
    package = check(path, full=False)
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != CPUS:
        raise ValueError('Run inside its prepared Slurm allocation with 20 CPUs per task')
    executable = shutil.which(os.environ.get('ORCA_COMMAND', 'orca'))
    if not executable:
        raise FileNotFoundError('ORCA executable is unavailable; load the configured module')
    for entry in package['entries'][index * PER_JOB:(index + 1) * PER_JOB]:
        work = MLIP / package['output'] / entry['role'] / 'artifacts' / entry['stem']
        work.parent.mkdir(parents=True, exist_ok=True)
        try:
            work.mkdir()
        except FileExistsError:
            print(f'Preserving {work}; existing attempts are never retried', flush=True)
            continue
        inp = work / f'{entry["stem"]}.inp'
        shutil.copy2(MLIP / entry['input_relative'], inp)
        output = work / f'{entry["stem"]}.out'
        # ORCA's driver launches MPI itself; do not prefix the driver with srun/mpirun.
        with output.open('x') as handle:
            result = subprocess.run([str(Path(executable).resolve()), str(inp)], cwd=work,
                                    stdout=handle, stderr=subprocess.STDOUT)
        (work / 'exitcode').write_text(f'{result.returncode}\n')
        print(f'{entry["stem"]}: exit {result.returncode}; retained all artifacts', flush=True)


def collect(path):
    from ase.io import read, write
    from orca_output_validation import Job, process
    package = check(path)
    frames = read(MLIP / package['dataset']['file'], index=':', format='extxyz')
    output = MLIP / package['output']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = output / 'collections' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    results, accepted = [], {role: [] for role in COUNTS}
    for entry, source in zip(package['entries'], frames, strict=True):
        work = output / entry['role'] / 'artifacts' / entry['stem']
        if not (work / 'exitcode').is_file() or (work / 'exitcode').read_text().strip() != '0':
            results.append({'index': entry['index'], 'role': entry['role'], 'status': 'rejected',
                            'issues': 'missing_or_nonzero_exitcode'})
            continue
        job = Job(entry['index'], entry['stem'], work / f'{entry["stem"]}.inp',
                  work / f'{entry["stem"]}.out', entry['charge'], entry['multiplicity'], entry['n_atoms'], METHOD)
        if digest(job.input_path) != entry['input_sha256']:
            raise ValueError(f'Raw input differs from prepared input: {job.input_path}')
        row, labelled = process(job)
        row.update(index=entry['index'], role=entry['role'])
        results.append(row)
        if labelled is not None:
            # Retain source frame/role metadata, including temporal partition provenance.
            info = dict(source.info)
            info.update(labelled.info)
            labelled.info = info
            accepted[entry['role']].append(labelled)
    for role, atoms in accepted.items():
        if atoms:
            write(destination / f'{role}.extxyz', atoms, format='extxyz')
    write_json(destination / 'validation.json', results)
    write_json(destination / 'summary.json', {'package': relative(path), 'package_sha256': digest(path),
        'accepted': {role: len(atoms) for role, atoms in accepted.items()},
        'energy_reference': package['energy_reference'], 'force_units': package['force_units'],
        'label_route': package['label_route'], 'training_rule': 'Never train on evaluation.extxyz'})
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--partition', choices=['general', 'small'], default='general')
    prep.add_argument('--walltime', default='24:00:00')
    prep.add_argument('--memory-gib', type=int, default=64)
    sub.add_parser('check')
    send = sub.add_parser('submit')
    send.add_argument('--dry-run', action='store_true')
    run = sub.add_parser('run-group')
    run.add_argument('--index', type=int, required=True)
    sub.add_parser('collect')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d|24:00:00', args.walltime) or args.memory_gib <= 0:
            parser.error('Viper requires positive memory and a walltime no longer than 24 hours')
        prepare(args)
        return
    path = resolve_package(args.package)
    if args.command == 'check':
        check(path)
        print('Verified 200 jobs, 2000 cluster inputs, formal charge/spin, provenance and frozen source hashes.')
    elif args.command == 'submit':
        submit(path, args.dry_run)
    elif args.command == 'run-group':
        run_group(path, args.index)
    else:
        collect(path)


if __name__ == '__main__':
    main()
