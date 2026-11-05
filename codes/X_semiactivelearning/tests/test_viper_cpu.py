"""Published-release CPU packaging, isolated job execution and fake Slurm checks."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('viper_cpu_tested', ROOT / 'dft/viper_cpu.py')
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


class ViperCPUTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.root = self.repo / ROOT.relative_to(REPO)
        self.root.mkdir(parents=True)
        self.patches = []
        for name, value in [('MLIP', self.repo), ('ROOT', self.root),
                            ('DATASET', self.root / 'partial_clusters_active.json'),
                            ('POINTER', self.root / 'cpu_dft_active.json'),
                            ('__file__', str(self.root / 'dft/viper_cpu.py'))]:
            p = patch.object(cpu, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(cpu.p10_batch, 'MLIP', self.repo)
        p.start()
        self.addCleanup(p.stop)
        for name in ['partial_clusters_active.json', 'site_profiles.json', 'dft/viper_cpu.py',
                     'production/campaign.py', 'production/p10_batch.py',
                     'production/orca_output_validation.py', 'production/workflow_policy.py']:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        target = self.root.parent / 'C6_wb97m_nodisp/prepare_data.py'
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT.parent / 'C6_wb97m_nodisp/prepare_data.py', target)
        release = cpu.load(cpu.DATASET)['dataset_directory']
        shutil.copytree(REPO / release, self.repo / release)
        args = type('Args', (), {'partition': 'general', 'walltime': '24:00:00', 'memory_gib': 64})()
        with redirect_stdout(io.StringIO()):
            self.path = cpu.prepare(args)
        self.package = cpu.load(self.path)
        self.bundle = self.path.parent

    def test_complete_package_geometry_roles_and_shell_syntax(self):
        package = cpu.check(self.path)
        self.assertEqual(len(list(self.bundle.glob('jobs/*.slurm'))), 200)
        self.assertEqual(len(list(self.bundle.glob('inputs/*.inp'))), 2000)
        self.assertEqual([e['role'] for e in package['entries'][1950:]], ['evaluation'] * 50)
        for path in [self.bundle / 'submitcpu.sh', *self.bundle.glob('jobs/*.slurm')]:
            subprocess.run(['bash', '-n', str(path)], check=True)
        text = (self.bundle / 'jobs/group_000.slurm').read_text()
        for setting in ['--partition=general', '--cpus-per-task=20', '--time=24:00:00', '--mem=64G']:
            self.assertIn(setting, text)
        self.assertNotIn('--array', text)

    def test_fake_sbatch_submits_exactly_200_and_prevents_duplicates(self):
        calls = self.repo / 'sbatch_calls'
        fake = self.repo / 'sbatch'
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_CALLS"\nprintf "12345\\n"\n')
        fake.chmod(0o755)
        env = {'PATH': f'{self.repo}:{os.environ["PATH"]}', 'FAKE_CALLS': str(calls)}
        with patch.dict(os.environ, env), redirect_stdout(io.StringIO()):
            cpu.submit(self.path, dry_run=True)
            self.assertFalse(calls.exists())
            cpu.submit(self.path)
            cpu.submit(self.path)
        lines = calls.read_text().splitlines()
        self.assertEqual(len(lines), 200)
        self.assertEqual(len(set(lines)), 200)
        self.assertTrue(all(line.startswith('--parsable ') for line in lines))
        self.assertEqual(len(list((self.repo / self.package['output'] / 'submissions').glob('*.json'))), 200)

    def test_failed_submission_stops_and_is_not_retried(self):
        fake = self.repo / 'sbatch'
        fake.write_text('#!/bin/bash\necho "test rejection" >&2\nexit 1\n')
        fake.chmod(0o755)
        with patch.dict(os.environ, {'PATH': f'{self.repo}:{os.environ["PATH"]}'}):
            with self.assertRaisesRegex(RuntimeError, 'Submission stopped'):
                cpu.submit(self.path)
            with self.assertRaisesRegex(RuntimeError, 'Unresolved submission'):
                cpu.submit(self.path)

    def test_tampered_input_is_rejected_before_sbatch(self):
        inp = self.bundle / 'inputs/cluster_0000.inp'
        inp.write_text(inp.read_text().replace('* xyz -1 1', '* xyz 0 1'))
        with patch.object(cpu.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
                cpu.submit(self.path)
            run.assert_not_called()

    def test_dataset_checksum_change_is_rejected(self):
        xyz = self.repo / self.package['dataset']['file']
        xyz.write_bytes(xyz.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            cpu.check(self.path)

    def test_missing_cpu_allocation_is_rejected(self):
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}):
            with self.assertRaisesRegex(ValueError, '20 CPUs'):
                cpu.run_group(self.path, 0)

    def test_runner_executes_ten_serial_clusters_and_preserves_attempts(self):
        fake = self.repo / 'fake_orca'
        fake.write_text('#!/bin/bash\necho "FAKE ORCA for runner plumbing only"\nexit 9\n')
        fake.chmod(0o755)
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '20', 'ORCA_COMMAND': str(fake)}), redirect_stdout(io.StringIO()):
            cpu.run_group(self.path, 0)
            cpu.run_group(self.path, 0)
        output = self.repo / self.package['output']
        attempts = list(output.glob('acquisition/artifacts/cluster_*'))
        self.assertEqual(len(attempts), 10)
        self.assertTrue(all((p / 'exitcode').read_text() == '9\n' for p in attempts))
        self.assertTrue(all((p / f'{p.name}.inp').is_file() for p in attempts))
        self.assertFalse(list(output.glob('evaluation/artifacts/*')))
        with redirect_stdout(io.StringIO()):
            cpu.collect(self.path)
        summary = next(output.glob('collections/*/summary.json'))
        self.assertEqual(cpu.load(summary)['accepted'], {'acquisition': 0, 'evaluation': 0})
        self.assertFalse(list(output.glob('collections/*/*.extxyz')))


if __name__ == '__main__':
    unittest.main()
