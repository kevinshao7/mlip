#!/usr/bin/env python3
"""Launch the isolated foundation/SWA recipe; --dry-run never trains."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MLIP = ROOT.parents[1]
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('XDG_CACHE_HOME', str(MLIP / 'outputsfull/.cache'))
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_plan(plan):
    for item in plan['inputs'].values():
        path = MLIP / item['path']
        if sha(path) != item['sha256']:
            raise ValueError(f'Frozen input hash mismatch: {path}')
    for relative, digest in plan['source_sha256'].items():
        if sha(MLIP / relative) != digest:
            raise ValueError(f'Prepared source changed; prepare a fresh bundle: {relative}')


def build_command(recipe, args, name, directory):
    command = recipe(args, name, directory)
    # Omit shared D2 optimizer/EMA/stage-two overrides: the installed MACE
    # parser and normal foundation-loading path own these defaults.
    omitted = ('--lr=', '--swa_lr=', '--ema_decay=',
               '--swa_energy_weight=', '--swa_forces_weight=', '--multiheads_finetuning=')
    command = [value for value in command if not value.startswith(omitted)]
    return command + ['--scheduler_patience=15', '--patience=30', '--eval_interval=4',
                      '--error_table=PerAtomMAE', '--save_cpu']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', choices=('s', 'm', 'l'), default='l')
    options = parser.parse_args()
    active = json.loads((ROOT / 'swa_foundation/active.json').read_text())
    plan = active['models'][options.model]
    verify_plan(plan)
    if os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Missing Slurm GPU assignment')
    sys.path.insert(0, str(MLIP / 'mace'))
    from mace.tools.arg_parser import build_default_arg_parser
    defaults = build_default_arg_parser()
    actual = {key: defaults.get_default(key) for key in plan['mace_defaults']}
    if actual != plan['mace_defaults']:
        raise ValueError('Installed MACE defaults changed; prepare a fresh bundle')
    path = ROOT.parent / 'D2_naive/trainmace.py'
    spec = importlib.util.spec_from_file_location('foundation_swa_recipe', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.SCRIPT_DIR = ROOT
    original_parse = recipe.parse_args
    def parse():
        args = original_parse()
        # Make launch metadata reflect upstream defaults, even though these
        # flags are intentionally omitted from the actual MACE command.
        for key, value in actual.items():
            setattr(args, key, value)
        args.scheduler_patience = 15
        args.patience = 30
        args.eval_interval = 4
        args.error_table = 'PerAtomMAE'
        args.save_cpu = True
        return args
    recipe.parse_args = parse
    original_build = recipe.build_train_argv
    recipe.build_train_argv = lambda *values: build_command(original_build, *values)
    original_log = recipe.append_experiment_log
    def log(args, name, directory, command):
        original_log(args, name, directory, command)
        with (ROOT / 'hyperparameters.txt').open('a') as stream:
            stream.write('\nFoundation/SWA launch details (seed123; supersedes shared seed3 path hints above):\n')
            stream.write(f'training_log={directory}/logs/{name}_run-123.log\n')
            stream.write(f'checkpoint_model_pattern={directory}/checkpoints/{name}_run-123_epoch-*.pt\n')
            stream.write(f'exported_model_directory={directory}/models\n')
            stream.write(f'compiled_model_if_successful={directory}/models/{name}_compiled.model\n')
            stream.write('source_plan=' + str(ROOT / 'swa_foundation/active.json') + '\n')
            stream.write('mace_defaults=' + json.dumps(actual, sort_keys=True) + '\n')
    recipe.append_experiment_log = log
    paths = {key: MLIP / item['path'] for key, item in plan['inputs'].items()}
    sys.argv = [str(path), '--name', f'polar1{options.model}_wb97m_no_disp_foundation_swa300',
                '--foundation-model', str(paths['foundation']),
                '--train-file', str(paths['train_file']), '--valid-file', str(paths['valid_file']),
                '--e0s', str(paths['e0s']), '--reference-method', 'wB97M/def2-TZVPD',
                '--atomic-numbers', '[1, 7, 8, 16]', '--runs-root', str(MLIP / plan['runs_root']),
                '--energy-weight', '10', '--forces-weight', '1000',
                '--batch-size', '2', '--valid-batch-size', '4', '--max-num-epochs', '300',
                '--start-swa', '150', '--swa', '--ema', '--seed', '123',
                '--default-dtype', 'float64', '--device', 'cuda', '--restart-latest',
                '--changed-parameter', 'recipe_baseline', '--change-note',
                'User-requested new foundation baseline: docs recipe, 300epochs/SWA150, audited dispersion-free wB97M E0s; upstream unspecified defaults, fixed P10 split; multiple factors differ from prior runs.']
    if options.dry_run:
        sys.argv.append('--dry-run')
    recipe.main()


if __name__ == '__main__':
    main()
