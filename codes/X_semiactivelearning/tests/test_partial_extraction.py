"""Committed-prefix safety checks for extraction from unfinished MD."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from extract_partial_clusters import copy_prefix, verify_workspace
import p10_batch as batch
from package_clusters import package


class SnapshotTests(unittest.TestCase):
    def test_uncommitted_tail_is_excluded_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.write_bytes(b'committed\nuncommitted partial frame')
            target = root / 'snapshot'
            copy_prefix(source, target, len(b'committed\n'))
            self.assertEqual(target.read_bytes(), b'committed\n')
            self.assertTrue(source.read_bytes().endswith(b'partial frame'))
            with self.assertRaises(FileExistsError):
                copy_prefix(source, target, 1)

    def test_short_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.write_bytes(b'short')
            with self.assertRaisesRegex(ValueError, 'shorter than committed'):
                copy_prefix(source, root / 'snapshot', 100)

    def test_snapshot_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / 'snapshot'
            snapshot.write_text('committed')
            batch.write_json(root / 'snapshot_manifest.json', {
                'sha256': {'snapshot': batch.sha(snapshot)}, 'source_sha256': {}})
            verify_workspace(root)
            snapshot.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
                verify_workspace(root)

    def package_fixture(self, root, duplicate=False):
        from ase import Atoms
        from ase.io import write
        batch.write_json(root / 'campaign_config.json', {
            'clusters': {'acquisition_count': 1, 'evaluation_count': 1}})
        summary = root / 'cycles/P10p0000GPa/analysis/summary.json'
        batch.write_json(summary, {})
        batch.write_json(root / 'snapshot_manifest.json', {'sha256': {}, 'source_sha256': {}})
        hashes = {}
        for role, frame in [('acquisition', 50), ('evaluation', 50 if duplicate else 0)]:
            atoms = Atoms('OH2', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            atoms.info.update(cluster_kind=role, source_trajectory_id='replica_00',
                              source_frame=frame, central_allowed_component=0,
                              charge=0, multiplicity=1)
            path = root / 'cycles/P10p0000GPa/clusters' / role / 'clusters.extxyz'
            path.parent.mkdir(parents=True)
            write(path, atoms, format='extxyz')
            hashes[role] = batch.sha(path)
        batch.write_json(root / 'cluster_quality.json', {
            'status': 'passed', 'total': 2, 'input_sha256': hashes,
            'analysis_sha256': batch.sha(summary),
            'snapshot_manifest_sha256': batch.sha(root / 'snapshot_manifest.json')})

    def test_packaging_retains_roles_and_refuses_overwrite(self):
        from ase.io import read
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.package_fixture(root)
            output = package(root)
            frames = read(output, index=':', format='extxyz')
            self.assertEqual([a.info['cluster_kind'] for a in frames], ['acquisition', 'evaluation'])
            with self.assertRaises(FileExistsError):
                package(root)

    def test_packaging_rejects_duplicate_source_across_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.package_fixture(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate source'):
                package(root)


if __name__ == '__main__':
    unittest.main()
