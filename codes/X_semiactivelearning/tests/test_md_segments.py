"""Tiny CPU fixtures verifying exact restart mechanics; no campaign MD runs."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read
from md_segments import save_restart,load_restart,restore_integrator,recover_output,staged_md,PAUSED_EXIT_CODE


def fixture():
    atoms=bulk('Ar','fcc',a=5.4,cubic=True)
    atoms.calc=LennardJones(epsilon=.01,sigma=3.4,rc=4.)
    MaxwellBoltzmannDistribution(atoms,temperature_K=100,rng=np.random.RandomState(7))
    return atoms


def dynamics(atoms):
    return IsotropicMTKNPT(atoms,timestep=.5*units.fs,temperature_K=100,
                           pressure_au=0.,tdamp=100*units.fs,pdamp=1000*units.fs)


class RestartTests(unittest.TestCase):
    def test_extended_npt_state_matches_uninterrupted_dynamics(self):
        atoms=fixture(); dyn=dynamics(atoms); dyn.run(6)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'restart.npz'; save_restart(path,dyn,{'model':'fixture'})
            restored=fixture(); meta,state=load_restart(path,restored,{'model':'fixture'})
            continued=dynamics(restored); restore_integrator(continued,state,meta)
            dyn.run(8); continued.run(8)
            np.testing.assert_allclose(restored.positions,atoms.positions,rtol=0,atol=1e-12)
            np.testing.assert_allclose(restored.get_momenta(),atoms.get_momenta(),rtol=0,atol=1e-12)
            np.testing.assert_allclose(restored.cell,atoms.cell,rtol=0,atol=1e-12)
            np.testing.assert_allclose(continued._thermostat._eta,dyn._thermostat._eta,rtol=0,atol=1e-12)
            np.testing.assert_allclose(continued._barostat._xi,dyn._barostat._xi,rtol=0,atol=1e-12)
            with self.assertRaisesRegex(ValueError,'setting changed'):
                load_restart(path,fixture(),{'model':'different'})

    def test_output_recovery_preserves_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'traj.xyz'; path.write_bytes(b'committed-extra')
            recover_output(path,9)
            self.assertEqual(path.read_bytes(),b'committed')
            archives=list(Path(temp).glob('*.uncommitted_*'))
            self.assertEqual(archives[0].read_bytes(),b'-extra')
            with self.assertRaisesRegex(ValueError,'truncated'):
                recover_output(path,99)

    def test_segment_resume_has_no_duplicate_frames_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            results=[]
            for label,budget in [('continuous','999'),('segmented','0')]:
                target=root/label; target.mkdir(); atoms=fixture()
                np.random.seed(17)
                with patch.dict(os.environ,{'MLIP_SEGMENT_SECONDS':budget}):
                    try:
                        staged_md(atoms,100,0.,atoms.calc,str(target/'prod.xyz'),str(target/'final.xyz'),10,40,T_thermo=0,pressure_ramp_ps=0,temperature_ramp_ps=0)
                    except SystemExit as exc:
                        self.assertEqual(exc.code,PAUSED_EXIT_CODE)
                        with patch.dict(os.environ,{'MLIP_SEGMENT_SECONDS':'999'}):
                            new_atoms=fixture()
                            staged_md(new_atoms,100,0.,new_atoms.calc,str(target/'prod.xyz'),str(target/'final.xyz'),10,40,T_thermo=0,pressure_ramp_ps=0,temperature_ramp_ps=0)
                frames=read(target/'prod.xyz',':')
                self.assertEqual(len(frames),4)
                status=json.loads((target/'production_status.json').read_text())
                self.assertTrue(status['complete']); self.assertEqual(status['step'],40)
                results.append(frames[-1])
            np.testing.assert_allclose(results[0].positions,results[1].positions,rtol=0,atol=1e-8)
            np.testing.assert_allclose(results[0].cell,results[1].cell,rtol=0,atol=1e-8)


if __name__=='__main__': unittest.main()
