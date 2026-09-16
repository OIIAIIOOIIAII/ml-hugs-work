"""Transform physical camera intrinsics to the frontend's two image grids."""
import numpy as np
from .rich_supervision import pixel_transform


def transformed_intrinsics(raw, width, height, size=512, pad_to=896):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.shape!=(3,3) or not np.isfinite(raw).all() or not np.allclose(raw[2],[0,0,1]):
        raise ValueError('Invalid pinhole intrinsics')
    if raw[0,0]<=0 or raw[1,1]<=0:raise ValueError('Invalid focal length')
    t=pixel_transform(width,height,size,pad_to)
    return np.asarray(t['raw_to_crop'])@raw,np.asarray(t['raw_to_pad'])@raw
