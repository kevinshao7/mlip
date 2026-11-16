"""Focused command-isolation checks for the foundation/SWA adapter."""
import importlib.util
from pathlib import Path
import tempfile
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('foundation_swa', ROOT / 'swa_foundation/launch_single_gpu.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class FoundationSWATests(unittest.TestCase):
    def test_submits_three_independent_model_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / 'sbatch'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=folder + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT / 'submit_swa_foundation.sh')],
                                    env=env, text=True, capture_output=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 3)
            for line, model in zip(lines, ('s', 'm', 'l')):
                self.assertIn('--model ' + model, line)
                self.assertIn('--job-name=X_P10_foundSWA_1' + model, line)
                self.assertNotIn('--dependency', line)
                self.assertNotIn('--array', line)

    def test_shared_overrides_do_not_leak_to_mace(self):
        shared = ['run_train.py', '--lr=0.0001', '--swa_lr=0.0003', '--ema_decay=0.99999',
                  '--swa_energy_weight=1', '--swa_forces_weight=100000',
                  '--multiheads_finetuning=False', '--foundation_model=frozen_foundation',
                  '--E0s=frozen_wb97m.json', '--swa', '--start_swa=150', '--max_num_epochs=300']
        command = launcher.build_command(lambda *args: shared, None, 'test', Path('/tmp'))
        for flag in ('lr', 'swa_lr', 'ema_decay', 'swa_energy_weight', 'swa_forces_weight',
                     'multiheads_finetuning', 'force_mh_ft_lr', 'clip_grad', 'weight_decay'):
            self.assertFalse(any(value.startswith('--' + flag + '=') for value in command), flag)
        for flag in ('--swa', '--start_swa=150', '--max_num_epochs=300', '--save_cpu',
                     '--scheduler_patience=15', '--patience=30', '--eval_interval=4',
                     '--error_table=PerAtomMAE', '--E0s=frozen_wb97m.json'):
            self.assertIn(flag, command)

    def test_frozen_input_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input'
            path.write_text('frozen')
            plan = {'inputs': {'foundation': {'path': str(path), 'sha256': launcher.sha(path)}},
                    'source_sha256': {}}
            launcher.verify_plan(plan)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen input hash mismatch'):
                launcher.verify_plan(plan)


if __name__ == '__main__':
    unittest.main()
