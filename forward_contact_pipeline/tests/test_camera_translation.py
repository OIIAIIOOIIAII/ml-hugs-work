import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.camera_translation import reproject_translation

class CameraTranslationTests(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[-.2,-.3,2.],[.2,.3,2.],[-.3,.2,2.],[.3,-.2,2.]])
        self.k=np.diag([100.,100.,1.])

    def test_equal_cameras_preserve_original_body_position(self):
        np.testing.assert_allclose(reproject_translation(self.v,self.k,self.k),0,atol=1e-10)

    def test_known_planar_focal_change_recovers_depth(self):
        new=self.k.copy();new[:2,:2]*=2
        np.testing.assert_allclose(reproject_translation(self.v,self.k,new),[0,0,2],atol=1e-10)

    def test_invalid_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'positive-depth'):
            reproject_translation(-self.v,self.k,self.k)

if __name__=='__main__':unittest.main()
