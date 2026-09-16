import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_geometry import native_mhmr_image,calibrated_images

class GeometryPreprocessingTests(unittest.TestCase):
 def test_sensor_calibration_translates_pixels_and_preserves_padding(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'rgb.png';raw=np.zeros((96,128,3),np.uint8);raw[44:53,60:69]=255;Image.fromarray(raw).save(p)
   k=np.array([[100.,0,64],[0,100,48],[0,0,1]])
   virtual=k.copy();virtual[0,2]+=8
   crop,high,A=calibrated_images(p,k,virtual,(128,96),128)
   self.assertEqual(tuple(crop.shape),(1,3,96,128))
   self.assertGreater(float(crop[0,0,48,72]),.99)
   self.assertLess(float(crop[0,0,48,60]),-.99)
   self.assertTrue(bool((high[:,:,:16,:]==-1).all()))
   np.testing.assert_allclose(A,[[1,0,8],[0,1,0],[0,0,1]])

 def test_native_mhmr_preserves_exact_padding_and_center_location(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'rgb.png';raw=np.zeros((640,1024,3),np.uint8);raw[310:331,502:523]=255;Image.fromarray(raw).save(p)
   high=native_mhmr_image(p,512,896)
   self.assertEqual(tuple(high.shape),(1,3,896,896))
   self.assertTrue(bool((high[:,:,:168,:]==-1).all()))
   self.assertTrue(bool((high[:,:,728:,:]==-1).all()))
   self.assertGreater(float(high[0,0,448,448]),.9)
   self.assertTrue(bool(high.isfinite().all()))

if __name__=='__main__':unittest.main()
