"""Isolated checks of scientific gates, label rejection and progression."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import read, write
import campaign
import autonomous
import workflow_policy
from orca_output_validation import Job, parse_input, process


class ForceGateTests(unittest.TestCase):
    def test_threshold_units_and_invalid_results(self):
        for value, expected in [(19.999, True), (20, False), (21, False), (float('nan'), False), (float('inf'), False), (-1, False), (True, False), (None, False)]:
            report = {'status':'complete', 'n_clusters':30, 'force_rmse_mev_per_angstrom':value}
            self.assertEqual(workflow_policy.passes_force_gate(report), expected)
        self.assertFalse(workflow_policy.passes_force_gate({'status':'unavailable', 'n_clusters':100, 'force_rmse_mev_per_angstrom':1}))
        self.assertFalse(workflow_policy.passes_force_gate({'status':'complete', 'n_clusters':29, 'force_rmse_mev_per_angstrom':1}))

    def test_temporal_partitions_are_stable(self):
        expected = ['evaluation', 'validation', 'training', 'training', 'training', 'training']
        self.assertEqual([workflow_policy.partition(i*50) for i in range(6)], expected)
        for i in range(300):
            self.assertEqual(workflow_policy.partition(i), expected[i//50])
        self.assertEqual(workflow_policy.partition(300), 'evaluation')


class DFTValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.atoms = Atoms('OH2', positions=[[0,0,0],[.95,0,0],[0,.95,0]])
        self.input = campaign.dft_input(self.atoms, 0, 1, 20, campaign.WB97M_REFERENCE_METHOD)
        self.inp = self.root/'water.inp'; self.inp.write_text(self.input)
        self.out = self.root/'water.out'
        self.output = '\n'.join(f'| {i+1}> {line}' for i,line in enumerate(self.input.splitlines())) + '''
Functional name .... wB97M-V exchange-correlation functional
Total Charge Charge .... 0
Multiplicity Mult .... 1
FINAL SINGLE POINT ENERGY -76.0
CARTESIAN GRADIENT
0 O : 0.001 0.0 0.0
1 H : -0.001 0.0 0.0
2 H : 0.0 0.0 0.0
ORCA TERMINATED NORMALLY
'''

    def result(self, text):
        self.out.write_text(text)
        return process(Job(0,'water',self.inp,self.out,0,1,3,campaign.WB97M_REFERENCE_METHOD))

    def test_route_roundtrip_and_units(self):
        self.assertEqual(parse_input(self.inp)[-1], campaign.WB97M_REFERENCE_METHOD)
        report, atoms = self.result(self.output)
        self.assertIsNotNone(atoms, report)
        self.assertEqual(atoms.info['charge'],0)
        self.assertFalse(atoms.pbc.any())
        self.assertAlmostEqual(atoms.info['REF_energy'], -76*27.211386245988)
        self.assertLess(atoms.arrays['REF_forces'][0,0],0)

    def test_rejects_dispersion_stale_charge_spin_and_failed_jobs(self):
        bad = [self.output+'\nDispersion correction -0.001\n',
               self.output.replace('Charge .... 0','Charge .... 1'),
               self.output.replace('Mult .... 1','Mult .... 3'),
               self.output.replace('ORCA TERMINATED NORMALLY','FAILED'),
               self.output.replace('0.950000000000','0.960000000000'),
               self.output+'\nFINAL SINGLE POINT ENERGY -75.0\n']
        for text in bad:
            report, atoms = self.result(text)
            self.assertIsNone(atoms, report)

    def test_refuses_input_charge_guess(self):
        with self.assertRaises(ValueError):
            campaign.dft_input(self.atoms,1,1,20,campaign.WB97M_REFERENCE_METHOD)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(campaign,'OUT',self.root); self.patch.start(); self.addCleanup(self.patch.stop)
        self.events = patch.object(campaign,'event'); self.events.start(); self.addCleanup(self.events.stop)
        campaign.write_json(self.root/'model_lineage.json', {'current_model':'polar-1-l','history':[]})
        self.path=campaign.cycle(10)/'dft/acquisition/processed.extxyz'; self.path.parent.mkdir(parents=True)

    def frame(self, i):
        atoms=Atoms('OH2',positions=[[0,0,0],[.95,0,0],[0,.95,0]])
        atoms.info.update(REF_energy=-2000., charge=0, spin=1, formal_charge_verified=True,
                          orca_input_charge=0, orca_output_charge=0, orca_input_multiplicity=1,
                          dft_functional_basis=campaign.WB97M_REFERENCE_METHOD,
                          source_pressure_gpa=10., source_trajectory_id='seed1',
                          source_frame=50 if i%5==0 else 100, source_center_component=i)
        atoms.arrays['REF_forces']=np.zeros((3,3))
        return atoms

    def prepare(self, frames):
        write(self.path,frames)
        args=campaign.parser().parse_args(['finetune','--pressure','10','--changed-parameter','acquisition_dataset','--change-note','isolated check'])
        with contextlib.redirect_stdout(io.StringIO()): campaign.command_finetune(args)
        return campaign.cycle(10)/'training'

    def test_stable_split_and_campaign_owned_launcher(self):
        frames=[self.frame(i) for i in range(30)]; data=self.prepare(frames)
        first=[a.info['source_center_component'] for a in read(data/'random_validation.extxyz',':')]
        data=self.prepare(frames+[self.frame(i) for i in range(30,40)])
        second=[a.info['source_center_component'] for a in read(data/'random_validation.extxyz',':')]
        self.assertTrue(set(first).issubset(second))
        command=(data/'launch_command.sh').read_text()
        self.assertIn('training_bridge.py',command)
        self.assertIn('e0s_wb97m.json',command)
        self.assertNotIn('--e0s=d4rev',command)

    def test_rejects_evaluation_leak_and_mixed_method(self):
        frames=[self.frame(i) for i in range(30)]
        frames[0].info['source_frame']=0
        with self.assertRaisesRegex(ValueError,'Held-out'): self.prepare(frames)
        frames[0].info['source_frame']=50
        frames[0].info['dft_functional_basis']='wB97M-D4rev def2-TZVPD'
        with self.assertRaisesRegex(ValueError,'Refusing to mix'): self.prepare(frames)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        patches=[patch.object(campaign,'OUT',self.root),patch.object(campaign,'event'),patch.object(autonomous,'note'),patch.object(workflow_policy,'require_method_verified')]
        for p in patches: p.start(); self.addCleanup(p.stop)
        self.model=self.root/'model.model'; self.model.write_text('saved model')
        campaign.write_json(self.root/'model_lineage.json', {'current_model':str(self.model),'history':[]})
        campaign.write_json(self.root/'method_verified.json',{})
        self.cdir=campaign.cycle(10)
        campaign.write_json(self.cdir/'md/complete.json',{})
        campaign.write_json(self.cdir/'analysis/summary.json',{'frames':2000,'cluster_eligible':True})
        campaign.write_json(self.cdir/'approval_acquisition.json',{})
        for kind in ['evaluation','acquisition']:
            path=self.cdir/'clusters'/kind/'clusters.extxyz'; path.parent.mkdir(parents=True); path.write_text('cluster')
            ddir=self.cdir/'dft'/kind; (ddir/'outputs').mkdir(parents=True)
            (ddir/'manifest.csv').write_text('stem\ncluster_000\n')
            (ddir/'outputs/cluster_000.exitcode').write_text('0')
            (ddir/'processed.extxyz').write_text('validated')
        acquired=self.cdir/'dft/acquisition/processed.extxyz'
        campaign.write_json(self.root/'model_lineage.json',{'current_model':str(self.model),'history':[{'acquired_sha256':{str(acquired.resolve()):campaign.sha(acquired)}}]})
        self.report={'dft_functional_basis':campaign.WB97M_REFERENCE_METHOD,'evaluation_scope':'reserved_temporal_blocks','status':'complete','parent_model':str(self.model),'model_sha256':campaign.sha(self.model), 'dataset_sha256':campaign.sha(self.cdir/'dft/evaluation/processed.extxyz'),'n_clusters':50,'force_rmse_mev_per_angstrom':19.9}

    def set_report(self): campaign.write_json(self.cdir/'dft/evaluation/parity_summary.json',self.report)

    def test_advance_exactly_five_only_after_pass(self):
        self.set_report()
        action=autonomous.next_action(); self.assertEqual(action,['record-pass','10'])
        autonomous.execute(action)
        self.assertEqual(autonomous.next_action(),['md','--pressure','15','--run'])

    def test_threshold_failure_collects_more_and_stale_model_rechecks(self):
        self.report['force_rmse_mev_per_angstrom']=20.; self.set_report()
        self.assertEqual(autonomous.next_action(),['clusters','--pressure','10','--kind','acquisition_r2'])
        self.model.write_text('changed checkpoint')
        self.assertEqual(autonomous.next_action()[-1],'--parity')

    def test_missing_or_small_eval_never_passes(self):
        self.report['n_clusters']=29; self.set_report()
        with self.assertRaisesRegex(ValueError,'Insufficient'): autonomous.next_action()
        self.report['n_clusters']=50; self.report['status']='unavailable'; self.set_report()
        with self.assertRaisesRegex(ValueError,'unavailable'): autonomous.next_action()

    def test_pending_slurm_does_not_relaunch_dft(self):
        (self.cdir/'dft/evaluation/outputs/cluster_000.exitcode').unlink()
        self.assertEqual(autonomous.next_action(),['await-dft','10','evaluation'])


class SitePackageTests(unittest.TestCase):
    def test_sites_preserve_inputs_and_valid_shell(self):
        import site_dft
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'outputsfull/X/run/dft/acquisition'
            stage.mkdir(parents=True)
            launch=root/'codes/X/dft_launches/run/acquisition'
            (launch/'inputs').mkdir(parents=True)
            (launch/'inputs/cluster_000.inp').write_text('geometry and charge unchanged')
            with patch.object(campaign,'MLIP',root):
                scripts=site_dft.prepare(stage,launch,[{'stem':'cluster_000'}],campaign.config()['dft'],campaign)
            self.assertEqual(len(scripts),2)
            for path in scripts:
                subprocess.run(['bash','-n',str(path)],check=True)
                self.assertIn('no automatic retry',path.read_text())
                self.assertIn('.partial',path.read_text())
            manifest=json.loads((launch/'site_package.json').read_text())
            self.assertEqual(manifest['input_sha256']['cluster_000'],campaign.sha(launch/'inputs/cluster_000.inp'))
            self.assertNotEqual(scripts[0].read_text(),scripts[1].read_text())

    def test_real_c6_pilot_loads_and_writes_site_packages(self):
        import method_pilot
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); script_root=root/'codes/X_semiactivelearning'
            script_root.mkdir(parents=True)
            relative=Path('outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/wb97m.extxyz')
            (root/relative).parent.mkdir(parents=True)
            shutil.copy2(campaign.MLIP/relative,root/relative)
            with patch.object(campaign,'ROOT',script_root), patch.object(campaign,'MLIP',root), patch.object(campaign,'OUT',root/'outputsfull/X_semiactivelearning/wb97m_step5'), patch.object(campaign,'event'), contextlib.redirect_stdout(io.StringIO()):
                method_pilot.prepare(None)
                pilot=campaign.OUT/'method_pilot'
                charge, spin, symbols, positions, method=parse_input(pilot/'inputs/pilot.inp')
                reference=read(pilot/'reference.extxyz')
                self.assertEqual(charge,campaign.formal_charge(symbols))
                self.assertEqual(spin,1)
                np.testing.assert_allclose(positions,reference.positions,atol=1e-10)
                self.assertEqual(method,campaign.WB97M_REFERENCE_METHOD)
                self.assertTrue((script_root/'dft_launches'/campaign.RUN_ID/'method_pilot/viper/submit_all.sh').exists())

    def test_round_ten_and_atomic_reference_audit(self):
        args=campaign.parser().parse_args(['clusters','--pressure','10','--kind','acquisition_r10'])
        self.assertEqual(args.kind,'acquisition_r10')
        from atomic_references import prepare
        result=prepare()
        self.assertEqual(set(json.loads(result.read_text())),{'1','7','8','16'})


if __name__ == '__main__': unittest.main()
