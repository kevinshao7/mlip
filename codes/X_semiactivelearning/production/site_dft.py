"""Prepare site-specific Slurm packages without changing scientific inputs."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(ddir, launch_root, records, cfg, campaign):
    profiles = json.loads((ROOT / 'site_profiles.json').read_text())
    group_size = int(cfg.get('slurm_group_size', 10))
    all_scripts = []
    for site, profile in profiles.items():
        destination = launch_root / site
        destination.mkdir(exist_ok=True)
        site_root = Path(profile['mlip_root'])
        relative = ddir.relative_to(campaign.MLIP)
        logdir = site_root / relative / 'slurm_logs'
        scripts = []
        for start in range(0, len(records), group_size):
            group = records[start:start+group_size]
            name = f'group_{start:04d}_{start+len(group)-1:04d}'
            path = destination / f'{name}.slurm'
            directives = ['#!/bin/bash -l', f'#SBATCH --job-name=X_{site}_{name}', '#SBATCH --nodes=1', f'#SBATCH --time={cfg["slurm_time"]}', f'#SBATCH --output={logdir}/{name}_%j.out', f'#SBATCH --error={logdir}/{name}_%j.err']
            if profile['partition']:
                directives.append(f'#SBATCH --partition={profile["partition"]}')
            if profile['exclude']:
                directives.append(f'#SBATCH --exclude={profile["exclude"]}')
            ranks = int(cfg['orca_nprocs'])
            if profile['layout'] == 'one_task_with_cpus':
                directives += ['#SBATCH --ntasks=1', f'#SBATCH --cpus-per-task={ranks}']
            else:
                directives += [f'#SBATCH --ntasks={ranks}', '#SBATCH --cpus-per-task=1']
            body = '''
set -euo pipefail
module purge
module load MODULE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MLIP_DIR="${MLIP_DIR:-SITE_ROOT}"
ORCA_COMMAND="${ORCA_COMMAND:-EXECUTABLE}"
if [[ "$ORCA_COMMAND" != /* ]]; then ORCA_COMMAND="$(command -v "$ORCA_COMMAND")"; fi
command -v mpirun >/dev/null
DFT_DIR="$MLIP_DIR/DFT_RELATIVE"
INPUT_DIR="$MLIP_DIR/INPUT_RELATIVE"
mkdir -p "$DFT_DIR/outputs" "$DFT_DIR/artifacts"
for stem in STEMS; do
    output="$DFT_DIR/outputs/${stem}.out"
    terminal="$DFT_DIR/outputs/${stem}.exitcode"
    if [[ -f "$terminal" || -f "$output" || -f "${output}.partial" ]]; then
        echo "Preserving existing result for $stem; no automatic retry"
        continue
    fi
    work="$DFT_DIR/artifacts/$stem"
    mkdir -p "$work"
    cp "$INPUT_DIR/${stem}.inp" "$work/"
    cd "$work"
    set +e
    "$ORCA_COMMAND" "$work/${stem}.inp" > "${output}.partial" 2>&1
    status=$?
    set -e
    mv "${output}.partial" "$output"
    printf '%s\\n' "$status" > "$terminal"
done
'''
            for key, value in {'MODULE': profile['module'], 'SITE_ROOT': str(site_root), 'EXECUTABLE': profile['orca_command'], 'DFT_RELATIVE': relative.as_posix(), 'INPUT_RELATIVE': (launch_root / 'inputs').relative_to(campaign.MLIP).as_posix(), 'STEMS': ' '.join(row['stem'] for row in group)}.items():
                body = body.replace(key, value)
            path.write_text('\n'.join(directives)+'\n'+body)
            scripts.append(path)
        submit = destination / 'submit_all.sh'
        submit.write_text('#!/bin/bash\nset -euo pipefail\n'
                          f'MLIP_DIR="${{MLIP_DIR:-{site_root}}}"\n'
                          f'mkdir -p "$MLIP_DIR/{relative}/slurm_logs" "$MLIP_DIR/{relative}/outputs"\n'
                          'cd "$(dirname "$0")"\n' + '\n'.join(f'sbatch {shlex.quote(path.name)}' for path in scripts)+'\n')
        all_scripts.extend(scripts)
    campaign.write_json(launch_root / 'site_package.json', {'functional_basis': cfg['functional_basis'], 'input_count': len(records), 'sites': profiles, 'outputs_relative_to_mlip': str(ddir.relative_to(campaign.MLIP)), 'input_sha256': {row['stem']: campaign.sha(launch_root / 'inputs' / f'{row["stem"]}.inp') for row in records}})
    return all_scripts


def submit(launch_root, site):
    subprocess.run(['bash', str(launch_root / site / 'submit_all.sh')], check=True)
