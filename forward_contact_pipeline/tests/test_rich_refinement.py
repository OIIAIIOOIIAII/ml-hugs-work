import sys
import unittest
from pathlib import Path
import numpy as np
import torch
try:
    import roma
except ImportError:
    roma = None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if roma is not None:
    from contact_streaming.rich_refinement import ResidualMLP, apply_residual

@unittest.skipIf(roma is None, 'Optional SMPL refinement tests require the GUSH3R environment with roma')
class RefinementContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.inputs=dict(rotation=roma.rotvec_to_rotmat(torch.randn(2,53,3)*.2),
                         shape=torch.randn(2,10),head=torch.tensor([[.1,.2,3.],[.3,-.2,5.]]))

    def test_initial_network_is_identity_with_finite_gradients(self):
        model=ResidualMLP(211,np.zeros(211),np.ones(211))
        out=model(torch.randn(2,211))
        self.assertEqual(float(out.abs().max()),0.)
        rotation,beta,head=apply_residual(self.inputs,out)
        torch.testing.assert_close(rotation,self.inputs['rotation'],rtol=0,atol=0)
        torch.testing.assert_close(beta,self.inputs['shape'],rtol=0,atol=0)
        torch.testing.assert_close(head,self.inputs['head'],rtol=0,atol=0)
        (rotation.sum()+beta.sum()+head.sum()).backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_large_predictions_respect_physical_bounds_and_hands(self):
        rotation,beta,head=apply_residual(self.inputs,torch.full((2,79),100.))
        angle=roma.rotmat_to_rotvec(rotation[:,:22]@self.inputs['rotation'][:,:22].transpose(-1,-2)).norm(dim=-1)
        self.assertLessEqual(float(angle.max()),1.20001)
        self.assertLessEqual(float((beta-self.inputs['shape']).abs().max()),3.00001)
        self.assertTrue(bool(((head-self.inputs['head']).norm(dim=-1)<=self.inputs['head'][:,2]*.25+1e-6).all()))
        torch.testing.assert_close(rotation[:,22:],self.inputs['rotation'][:,22:],rtol=0,atol=0)
        torch.testing.assert_close(torch.det(rotation),torch.ones(2,53),atol=1e-5,rtol=0)

    def test_labels_do_not_affect_reconstruction_parameters(self):
        residual=torch.randn(2,79)*.01
        before=apply_residual(self.inputs,residual)
        after=apply_residual(dict(self.inputs,target=torch.full((2,10475,3),float('nan')),y=torch.randn(2,79)),residual)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
