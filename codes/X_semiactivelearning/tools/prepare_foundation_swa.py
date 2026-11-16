"""Freeze inputs for the separate foundation/SWA baseline; never submit or train."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MLIP = ROOT.parents[1]
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(ROOT / 'swa_foundation'))
from launch_single_gpu import sha


def prepare_model(model, stamp):
    from mace.tools.arg_parser import build_default_arg_parser
    selected = json.loads((ROOT / 'low_lr_active.json').read_text())['models'][model]
    foundation = Path(selected['arguments']['foundation_model'])
    if foundation.name != 'MACEPOLAR1' + model.upper() + 'model':
        raise ValueError('Foundation size does not match requested model')
    frozen = selected['files']
    canonical = ROOT / 'atomic_references/e0s_wb97m.json'
    if sha(MLIP / frozen['e0s']['path']) != sha(canonical):
        raise ValueError('E0s differ from audited dispersion-free wB97M references')
    out = MLIP / 'outputsfull/X_semiactivelearning' / ('p10_foundation_swa300_' + stamp) / ('polar1' + model)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {'foundation': {'path': str(foundation.relative_to(MLIP)), 'sha256': sha(foundation)}}
    for key, item in frozen.items():
        source = MLIP / item['path']
        if sha(source) != item['sha256']:
            raise ValueError(f'Frozen dataset hash mismatch: {source}')
        target = out / source.name
        shutil.copy2(source, target)
        inputs[key] = {'path': str(target.relative_to(MLIP)), 'sha256': sha(target)}
    parser = build_default_arg_parser()
    keys = ('lr', 'swa_lr', 'ema_decay', 'swa_energy_weight', 'swa_forces_weight',
            'weight_decay', 'clip_grad', 'optimizer', 'scheduler', 'lr_factor',
            'force_mh_ft_lr', 'num_workers')
    sources = [ROOT / 'swa_foundation/launch_single_gpu.py', ROOT / 'tools/prepare_foundation_swa.py',
               ROOT / 'finetune_swa_foundation_dais.slurm', ROOT / 'submit_swa_foundation.sh',
               ROOT.parent / 'D2_naive/trainmace.py',
               MLIP / 'mace/mace/tools/arg_parser.py', MLIP / 'mace/mace/tools/scripts_utils.py',
               MLIP / 'mace/mace/tools/model_script_utils.py', MLIP / 'mace/mace/cli/run_train.py',
               MLIP / 'mace/mace/tools/train.py']
    plan = {'created_utc': stamp, 'inputs': inputs,
            'runs_root': str((out / 'runs').relative_to(MLIP)),
            'mace_defaults': {key: parser.get_default(key) for key in keys},
            'source_sha256': {str(path.relative_to(MLIP)): sha(path) for path in sources},
            'evaluation_held_out': 50, 'train_count': 950, 'validation_count': 1000,
            'max_num_epochs': 300, 'start_swa': 150,
            'initialization': f'original cached polar-1-{model} foundation; no trained checkpoints',
            'split': 'unchanged frozen P10 train/validation/evaluation partitions'}
    text = json.dumps(plan, indent=2) + '\n'
    (out / 'manifest.json').write_text(text)
    return plan


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    plans = {model: prepare_model(model, stamp) for model in ('s', 'm', 'l')}
    active = {'created_utc': stamp, 'models': plans}
    (ROOT / 'swa_foundation/active.json').write_text(json.dumps(active, indent=2) + '\n')
    print('Prepared polar-1-s, polar-1-m, polar-1-l: ' + stamp)


if __name__ == '__main__':
    main()
