"""Contracts for held-out coverage and the learnable kinematic residual model."""
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from contact_streaming.rich_kinematic import KinematicResidualNet, BODY_PARENTS
from contact_streaming.rich_sensor_intrinsics import transformed_intrinsics
from contact_streaming.rich_supervision import pixel_transform
try:
    import roma
    import smplx
except ImportError:
    HAS_GEOMETRY = False
else:
    HAS_GEOMETRY = True
    from train_rich_coverage_study import assemble, features, make_model, check_plan_hash


class KinematicContracts(unittest.TestCase):
    def test_identity_can_learn_and_reaches_visual_features(self):
        torch.manual_seed(19)
        model = KinematicResidualNet(2003, np.zeros(2003), np.ones(2003), dropout=0)
        x = torch.randn(8, 2003)
        self.assertEqual(tuple(model(x).shape), (8, 79))
        self.assertEqual(float(model(x).abs().max()), 0.)
        target = torch.randn(8, 79)*.1
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(3):
            opt.zero_grad()
            ((model(x)-target)**2).mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            opt.step()
        self.assertGreater(float(model.visual[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.joint_identity.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model(x).abs().max()), 0)

    def test_adjacency_is_connected_and_respects_parent_tree(self):
        model = KinematicResidualNet(2003, np.zeros(2003), np.ones(2003))
        a = model.adjacency
        torch.testing.assert_close(a.sum(1), torch.ones(22))
        self.assertTrue(torch.equal(a > 0, a.T > 0))
        for joint, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                self.assertGreater(float(a[joint, parent]), 0)
        self.assertEqual(int((a > 0).sum()), 22 + 2*21)
        self.assertTrue(bool((torch.linalg.matrix_power(a, 21) > 0).all()))


class SensorIntrinsicsContracts(unittest.TestCase):
    def test_projects_to_identical_pixels_after_landscape_portrait_transform(self):
        k=np.array([[3200.,0,1990.],[0,3100.,1490.],[0,0,1.]])
        points=np.array([[.2,.3,3.],[-.6,-.2,6.]])
        for width,height in [(4112,3008),(3008,4112)]:
            crop,pad=transformed_intrinsics(k,width,height)
            t=pixel_transform(width,height,512,896)
            raw=points@k.T;raw=raw/raw[:,2:3]
            for actual,key in [(crop,'raw_to_crop'),(pad,'raw_to_pad')]:
                projected=points@actual.T;projected/=projected[:,2:3]
                np.testing.assert_allclose(projected,raw@np.array(t[key]).T,atol=1e-10)

    def test_invalid_intrinsics_rejected(self):
        for k in [np.zeros((3,3)),np.eye(4),np.full((3,3),np.nan)]:
            with self.assertRaises(ValueError):transformed_intrinsics(k,4112,3008)


@unittest.skipUnless(HAS_GEOMETRY, 'Coverage trainer needs optional GUSH3R geometry dependencies')
class CoverageContracts(unittest.TestCase):
    @staticmethod
    def rows(spec):
        rows = [(sub,cam,clip,f's/{seq}/cam_{cam:02}/{frame}',seq,frame)
                for sub,cam,clip,seq,frames in spec for frame in frames]
        return dict(subject=np.array([r[0] for r in rows]), clip=np.array([r[2] for r in rows]),
                    sample_id=np.array([r[3] for r in rows]), sequence=np.array([r[4] for r in rows]),
                    frame_id=np.array([r[5] for r in rows]), pose=np.zeros((len(rows),211)),
                    token=np.ones((len(rows),1792)))

    def test_nested_budgets_holdout_and_whole_clip_dedup(self):
        old = self.rows([('000',0,0,'a',[1,2]), ('003',0,1,'t',[1,2]), ('000',4,2,'a',[1,2])])
        new = self.rows([('000',0,0,'a',[2,3]), ('001',1,1,'b',[1,2]),
                         ('003',4,2,'t',[3,4]), ('020',4,3,'u',[3,4])])
        budgets,tune,dropped = assemble(old,new)
        self.assertEqual(dropped,[0])
        self.assertEqual(len(budgets['small']['pose']),2)
        self.assertEqual(len(budgets['expanded']['pose']),4)
        self.assertEqual(set(tune['subject']),{'003','020'})
        self.assertEqual(set(budgets['expanded']['clip']),{0,100001})
        self.assertTrue(set(budgets['small']['sample_id']) <= set(budgets['expanded']['sample_id']))
        self.assertFalse(any(s.endswith('/3') for s in budgets['expanded']['sample_id']))

    def test_model_roundtrip_and_no_label_features(self):
        torch.manual_seed(7)
        data = self.rows([('000',0,0,'a',[1,2])])
        x = features(data)
        data.update(target=np.full((2,100,3),np.nan), y=np.full((2,79),np.nan))
        np.testing.assert_array_equal(x,features(data))
        for architecture in ['mlp','kinematic']:
            model = make_model(architecture,x,device='cpu',dropout=0)
            with torch.no_grad():
                for p in model.parameters(): p.add_(torch.randn_like(p)*.01)
            buffer = io.BytesIO()
            torch.save(model.state_dict(),buffer); buffer.seek(0)
            state = torch.load(buffer,weights_only=True)
            clone = make_model(architecture,x,state=state,device='cpu',dropout=0)
            torch.testing.assert_close(model(torch.tensor(x).float()),clone(torch.tensor(x).float()),rtol=0,atol=0)

    def test_plan_provenance_rejects_unplanned_data(self):
        check_plan_hash({'provenance':{'some/development_plan.json':'abc','data.npz':'xyz'}},'abc')
        with self.assertRaises(ValueError):
            check_plan_hash({'provenance':{'some/development_plan.json':'wrong'}},'abc')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
