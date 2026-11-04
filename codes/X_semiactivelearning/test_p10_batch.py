"""Preparation-only checks; no Slurm jobs or real MD/DFT calculations run."""
import argparse
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
import campaign
import p10_batch as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.code=self.root/'codes/X_semiactivelearning'
        self.code.mkdir(parents=True)
        for name in ['campaign.py','p10_batch.py','workflow_policy.py','campaign_config.json','method_pilot.py','orca_output_validation.py','site_dft.py','site_profiles.json','cluster_quality.py']:
            shutil.copy2(batch.ROOT/name,self.code/name)
        profile=Path(json.loads((self.code/'campaign_config.json').read_text())['profile_csv'])
        target=(self.code/profile).resolve(); target.parent.mkdir(parents=True)
        shutil.copy2((batch.ROOT/profile).resolve(),target)
        template=self.root/'codes/B_conditionsproduction/NPTMACEproduction_base.py'
        template.parent.mkdir(); shutil.copy2(batch.ROOT.parent/'B_conditionsproduction/NPTMACEproduction_base.py',template)
        for name,value in [('ROOT',self.code),('MLIP',self.root),('ACTIVE_PLAN',self.code/'p10_batch_active.json')]:
            p=patch.object(batch,name,value); p.start(); self.addCleanup(p.stop)
        for name in ['OUT','RUN_ID']:
            p=patch.object(campaign,name,getattr(campaign,name)); p.start(); self.addCleanup(p.stop)
        with patch.object(batch,'notebook'), contextlib.redirect_stdout(io.StringIO()):
            batch.prepare(argparse.Namespace(new_batch=False))
        self.plan=batch.load_plan()

    def test_exact_budget_seeds_and_single_gpu_array(self):
        self.assertEqual(len({r['seed'] for r in self.plan['replicas']}),10)
        self.assertEqual(self.plan['acquisition_clusters']+self.plan['evaluation_clusters'],2000)
        code=self.root/self.plan['code_relative']
        for path in code.iterdir():
            if path.suffix in ['.sh','.slurm']:
                subprocess.run(['bash','-n',str(path)],check=True)
        md=(code/'md_10_replicas_dais.slurm').read_text()
        self.assertIn('#SBATCH --array=0-9',md)
        self.assertIn('--gres=gpu:h200:1',md)
        self.assertIn('--time=24:00:00',md)
        self.assertIn('--mem=64G',md)
        self.assertEqual(self.plan['md']['production_duration_ps'],176)
        self.assertEqual(self.plan['model_mode'],'foundation')
        self.assertNotIn('CUDA_VISIBLE_DEVICES=',md)
        self.assertIn('#SBATCH --array=0-199%20',(code/'dft_2000_viper.slurm').read_text())

    def test_missing_model_and_wrong_foundation_block(self):
        with self.assertRaises(FileNotFoundError): batch.binding(self.plan)
        model=self.root/'selected.model'; model.write_text('test fixture, never loaded')
        metadata=self.root/'run_metadata.json'; metadata.write_text(json.dumps({'arguments':{'foundation_model':'polar-1-s'}}))
        with self.assertRaisesRegex(ValueError,'foundation polar-1-l'):
            batch.bind_model(argparse.Namespace(model=model,metadata=metadata),self.plan)

    def test_checkpoint_binding_and_hash_lock(self):
        model=self.root/'selected.model'; model.write_text('test fixture, never loaded')
        with patch.object(batch,'notebook'),patch.object(batch,'download_foundation',return_value=(model,{'alias':'polar-1-l','url':'fixture-only'})),contextlib.redirect_stdout(io.StringIO()):
            batch.cache_foundation(self.plan)
        frozen,record=batch.binding(self.plan)
        self.assertEqual(record['sha256'],batch.sha(model))
        frozen.write_text('altered')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'): batch.binding(self.plan)

    def test_source_and_configuration_are_frozen(self):
        source=self.code/'campaign.py'; source.write_text(source.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'source changed'): batch.configure(self.plan)

    def test_preflight_recomputes_formal_charge(self):
        atoms=Atoms('OH2',positions=[[0,0,0],[.95,0,0],[0,.95,0]])
        path=self.root/'test.inp'; path.write_text(campaign.dft_input(atoms,0,1,20,campaign.WB97M_REFERENCE_METHOD))
        entry={'input_relative':'test.inp','input_sha256':batch.sha(path),'charge':'0','multiplicity':'1','n_atoms':'3'}
        self.assertEqual(batch.validate_input(entry),path)
        entry['charge']='1'
        with self.assertRaisesRegex(ValueError,'charge mismatch'): batch.validate_input(entry)
        entry['charge']='0'; path.write_text(path.read_text().replace('* xyz 0 1','* xyz 1 1')); entry['input_sha256']=batch.sha(path)
        with self.assertRaisesRegex(ValueError,'charge mismatch'): batch.validate_input(entry)

    def test_replica_code_preparation_does_not_run_md(self):
        import ast
        batch.configure(self.plan)
        patches=[patch.object(campaign,'ROOT',self.code),patch.object(campaign,'MLIP',self.root),
                 patch.object(campaign,'read_profile',return_value=(.344,1920.)),
                 patch.object(campaign,'current_parent',return_value=str(self.root/'fine_tuned.model')),
                 patch.object(campaign,'event')]
        with contextlib.ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'fixture','CUDA_VISIBLE_DEVICES':'GPU-fixture'}))
            with patch.object(campaign.subprocess,'run') as runner, contextlib.redirect_stdout(io.StringIO()):
                for replica in self.plan['replicas'][:2]:
                    args=campaign.parser().parse_args(['md','--pressure','10','--replica-id',replica['id'],'--seeds',str(replica['seed'])])
                    args.func(args)
                for call in runner.call_args_list:
                    self.assertEqual(call.args[0], "lscpu", "Only a library CPU-capability query is permitted during preparation")
            for replica in self.plan['replicas'][:2]:
                directory=campaign.cycle(10)/'md'/replica['id']
                manifest=json.loads((directory/'manifest.json').read_text())
                codepath=Path(manifest['replicas'][0]['launcher'])
                text=codepath.read_text(); ast.parse(text)
                self.assertEqual(manifest['production_duration_ps'],176)
                self.assertEqual(manifest['total_steps'],360000)
                self.assertEqual(manifest['save_interval_steps'],176)
                self.assertIn(f"RUN_SEED = {replica['seed']}",text)
                self.assertIn('torch.manual_seed(RUN_SEED)',text)
                self.assertIn('GPU-fixture',text)
                self.assertIn(str(directory),text)
                self.assertIn('model_neighbor_cutoff_angstrom',text)
                self.assertFalse((directory/'complete.json').exists())

    def test_configuration_transfers_with_code(self):
        out=self.root/self.plan['output_relative']
        (out/'campaign_config.json').unlink()
        batch.configure(self.plan)
        self.assertEqual(batch.sha(out/'campaign_config.json'),self.plan['config_sha256'])


if __name__=='__main__': unittest.main()
