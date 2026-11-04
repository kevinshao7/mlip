"""Regression checks for complete-molecule extraction at PBC image boundaries."""
import unittest
from pathlib import Path
import numpy as np
from ase.geometry import find_mic
from unittest.mock import patch
import campaign
from workflow_species import Frame
from cluster_quality import molecule_images, audit_cluster


class PeriodicClusterTests(unittest.TestCase):
    def frame(self,centres):
        positions=[]; symbols=[]; components=[]
        for x,y,z in centres:
            first=len(symbols); symbols.extend(['O','H','H'])
            positions.extend([[x,y,z],[x+.85,y+.3,z],[x-.2,y+.87,z]])
            components.append([first,first+1,first+2])
        return Frame(tuple(symbols),np.array(positions)%10,np.eye(3)*10,''),components

    def test_unwraps_whole_molecule_crossing_half_cell_surface(self):
        frame,comps=self.frame([(0,0,0),(4.7,2,0),(2,4,0),(-3,0,0),(0,-3,0),(0,0,3),(0,0,-3)])
        images,centres,error=molecule_images(frame,comps,0,range(len(comps)))
        self.assertLess(error,1e-8)
        for ci,indices in enumerate(comps):
            for left in range(3):
                for right in range(3):
                    _,source=find_mic(frame.positions[indices[left]]-frame.positions[indices[right]],frame.cell,pbc=True)
                    self.assertAlmostEqual(np.linalg.norm(images[ci][left]-images[ci][right]),source)
        # The old atomwise centre-anchor wrapping tears this neighbour's O-H bond.
        old,_=find_mic(frame.positions[comps[1]]-frame.positions[0],frame.cell,pbc=True)
        self.assertGreater(np.linalg.norm(old[1]-old[0]),8)
        self.assertLess(np.linalg.norm(images[1][1]-images[1][0]),1)

    def test_worker_selects_five_nearest_and_preserves_molecules(self):
        frame,comps=self.frame([(0,0,0),(4.7,2,0),(2,4,0),(-3,0,0),(0,-3,0),(0,0,3),(0,0,-3)])
        row={'trajectory_id':'replica_00','frame':'50','center_component':'0','source_pressure_gpa':10,'cluster_kind':'acquisition'}
        with patch.object(campaign,'fast_molecular_components',return_value=comps):
            result=campaign.build_cluster_worker((0,row,frame,Path('fixture.xyz'),('H2O',),6,False))
        atoms=result[1]; report=audit_cluster(atoms)
        self.assertEqual(len(atoms),18)
        _,centres,_=molecule_images(frame,comps,0,range(len(comps)))
        expected=[0]+sorted(range(1,len(comps)),key=lambda i:np.linalg.norm(centres[i]-centres[0]))[:5]
        for offset,ci in enumerate(expected):
            np.testing.assert_allclose(atoms.positions[offset*3:offset*3+3].mean(axis=0),centres[ci],atol=1e-8)
        self.assertEqual(report['charge'],0)
        self.assertLess(report['max_internal_distance_error_angstrom'],1e-8)
        self.assertFalse(atoms.pbc.any())
        self.assertEqual(atoms.info['molecule_atom_counts'],[3]*6)
        self.assertEqual(atoms.info['selected_formulas'],';'.join(['H2O']*6))

    def test_sampling_respects_actual_start_frame_and_stride(self):
        from workflow_policy import partition_frames
        evaluation=partition_frames(180,start=100,stride=2,evaluation=True)
        acquisition=partition_frames(180,start=100,stride=2,evaluation=False)
        self.assertEqual(evaluation,list(range(300,350,2)))
        self.assertFalse(set(evaluation)&set(acquisition))
        self.assertEqual(sorted(evaluation+acquisition),list(range(100,460,2)))

    def test_invalid_quality_metadata_is_rejected(self):
        frame,comps=self.frame([(0,0,0),(3,0,0),(0,3,0),(-3,0,0),(0,-3,0),(0,0,3)])
        row={'trajectory_id':'replica_00','frame':'50','center_component':'0','source_pressure_gpa':10,'cluster_kind':'acquisition'}
        with patch.object(campaign,'fast_molecular_components',return_value=comps):
            atoms=campaign.build_cluster_worker((0,row,frame,Path('fixture.xyz'),('H2O',),6,False))[1]
        atoms.info['charge']=1
        with self.assertRaisesRegex(ValueError,'charge'): audit_cluster(atoms)
        atoms.info['charge']=0; atoms.info['whole_molecules_preserved']=False
        with self.assertRaisesRegex(ValueError,'whole-molecule'): audit_cluster(atoms)


if __name__=='__main__': unittest.main()
