"""Numerical geometry experiment helpers; no ground-truth input to the model."""
from __future__ import annotations
import numpy as np
from .rich_supervision import pixel_transform


def native_mhmr_image(path, size, target_size):
    """Decode raw RGB directly to MHMR at the *same* field of view as CUT3R.

    Avoid the released raw -> 512 -> 896 resampling. PIL's box coordinates
    describe pixel edges, matching our pixel-center affine convention.
    """
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as raw:
        if int(raw.getexif().get(274,1))!=1: raise ValueError('Require audited identity EXIF')
        raw=raw.convert('RGB'); width,height=raw.size
        t=pixel_transform(width,height,size,pad_to=target_size)
        rw,rh=t['resized_size_wh']; cw,ch=t['crop_size_wh']
        scale=min(target_size/cw,target_size/ch); nw,nh=int(cw*scale),int(ch*scale)
        left,top,right,bottom=t['crop_xyxy']
        resized=raw.resize((nw,nh),Image.Resampling.LANCZOS,
            box=(left*width/rw,top*height/rh,right*width/rw,bottom*height/rh))
        canvas=ImageOps.expand(resized,border=((target_size-nw)//2,(target_size-nh)//2,
            target_size-nw-(target_size-nw)//2,target_size-nh-(target_size-nh)//2),fill=0)
        array=np.asarray(canvas,dtype=np.float32).copy()/127.5-1
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0)


def reconstruct_mesh(res, layer):
    """Head-relative SMPL-X in predicted camera space, matching HumanGS convention."""
    import torch
    import roma
    rot=res['smpl_rotmat'][0]
    if rot.shape[0]==0: return None
    rv=roma.rotmat_to_rotvec(rot)
    kwargs=dict(betas=res['smpl_shape'][0],expression=res['smpl_expression'][0],
        global_orient=rv[:,0],body_pose=rv[:,1:22].reshape(-1,63),
        left_hand_pose=rv[:,22:37].reshape(-1,45),right_hand_pose=rv[:,37:52].reshape(-1,45),
        jaw_pose=rv[:,52],leye_pose=torch.zeros_like(rv[:,52]),reye_pose=torch.zeros_like(rv[:,52]))
    body=layer(**kwargs)
    offset=res['smpl_transl'][0]-body.joints[:,15]
    return body.vertices+offset[:,None],body.joints[:,:55]+offset[:,None]


def reconstruct_humangs_geometry(res, mesh, c2w):
    """Exactly reproduce the released HumanGS world-space mesh/head FK path.

    The alternative standard SMPL-X joint return has millimeter differences
    from HumanGS's separate FK implementation; never silently mix baselines.
    """
    import torch
    import roma
    rot=res['smpl_rotmat'][0]
    if not len(rot): return None
    rv=roma.rotmat_to_rotvec(rot)
    p=dict(betas=res['smpl_shape'][0],expr=res['smpl_expression'][0],
        root_pose=roma.rotmat_to_rotvec(c2w[:3,:3]@rot[:,0]),
        body_pose=rv[:,1:22],lhand_pose=rv[:,22:37],rhand_pose=rv[:,37:52],
        jaw_pose=rv[:,52],leye_pose=torch.zeros_like(rv[:,52]),reye_pose=torch.zeros_like(rv[:,52]),
        trans=res['smpl_transl'][0]@c2w[:3,:3].T+c2w[:3,3])
    _,head,_=mesh.get_target_transform(p)
    layer=mesh.layer['neutral']
    out=layer(betas=p['betas'],expression=p['expr'],global_orient=p['root_pose'],
        body_pose=p['body_pose'],left_hand_pose=p['lhand_pose'],right_hand_pose=p['rhand_pose'],
        jaw_pose=p['jaw_pose'],leye_pose=p['leye_pose'],reye_pose=p['reye_pose'])
    world=out.vertices+(p['trans']-head)[:,None]
    world_joints=out.joints[:,:55]+(p['trans']-head)[:,None]
    return (world-c2w[:3,3])@c2w[:3,:3],(world_joints-c2w[:3,3])@c2w[:3,:3],world


def calibrated_images(path, K_raw, K_virtual, crop_size_wh, target_size):
    """Resample RGB into the model's virtual camera using sensor calibration.

    Camera calibration is extra sensor metadata, not a body/contact label.
    Optical axes and camera origin are unchanged; only intrinsics are changed.
    """
    import cv2
    import torch
    from PIL import Image
    with Image.open(path) as im:
        if int(im.getexif().get(274,1))!=1: raise ValueError('Require identity EXIF')
        raw=np.asarray(im.convert('RGB'))
    width,height=crop_size_wh
    A=np.asarray(K_virtual).reshape(3,3)@np.linalg.inv(np.asarray(K_raw))
    cropped=cv2.warpAffine(raw,A[:2],(width,height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    scale=min(target_size/height,target_size/width);nw,nh=int(width*scale),int(height*scale)
    sx,sy=nw/width,nh/height;pl,pt=(target_size-nw)//2,(target_size-nh)//2
    B=np.array([[sx,0,(sx-1)/2+pl],[0,sy,(sy-1)/2+pt],[0,0,1.]])
    high=cv2.warpAffine(raw,(B@A)[:2],(target_size,target_size),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    # Preserve exactly the same black padding used by MHMR's released loader.
    high[:pt]=0;high[pt+nh:]=0;high[:,:pl]=0;high[:,pl+nw:]=0
    def tensor(x):return torch.from_numpy(x.astype(np.float32)/127.5-1).permute(2,0,1).unsqueeze(0)
    return tensor(cropped),tensor(high),A.tolist()
