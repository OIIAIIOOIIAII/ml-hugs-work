"""Frozen all-person RGB features; annotation association is a separate final step."""
import gc,hashlib,io,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from .rich_assets import sha256
from .rich_geometry import reconstruct_humangs_geometry
from .rich_sensor_intrinsics import transformed_intrinsics
from .rich_supervision import pixel_transform,vertex_normals
from .rich_full import LabelStore,match_targets,scene_patch,INPUT_KEYS


def prepare_frame(rich,frame,index=0):
    """One verified decode, with pixel-equivalent released 512/native896 preprocessing."""
    from dust3r.utils.image import ImgNorm
    from dust3r.utils.geometry import get_camera_parameters
    payload=(Path(rich)/frame['image']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=frame['image_sha256']:raise ValueError('Image changed')
    with Image.open(io.BytesIO(payload)) as raw:
        if int(raw.getexif().get(274,1))!=1 or list(raw.size)!=frame['raw_size_wh']:raise ValueError('Image header changed')
        raw=raw.convert('RGB');w,h=raw.size;t=pixel_transform(w,h,512,896)
        rw,rh=t['resized_size_wh'];cw,ch=t['crop_size_wh'];left,top,right,bottom=t['crop_xyxy']
        low=ImgNorm(raw.resize((rw,rh),Image.Resampling.LANCZOS).crop((left,top,right,bottom)))[None]
        scale=min(896/cw,896/ch);pw,ph=int(cw*scale),int(ch*scale)
        high=raw.resize((pw,ph),Image.Resampling.LANCZOS,box=(left*w/rw,top*h/rh,right*w/rw,bottom*h/rh))
        high=ImageOps.expand(high,border=((896-pw)//2,(896-ph)//2,896-pw-(896-pw)//2,896-ph-(896-ph)//2),fill=0)
        high=torch.from_numpy(np.asarray(high,dtype=np.float32).copy()/127.5-1).permute(2,0,1)[None]
    return dict(img=low,img_mhmr=high,ray_map=torch.full((1,6,ch,cw),torch.nan),true_shape=torch.tensor([[ch,cw]],dtype=torch.int32),
        idx=index,instance=str(index),camera_pose=torch.eye(4)[None],
        camera_intrinsics=get_camera_parameters(max(cw,ch),p_x=cw/(2*max(cw,ch)),p_y=ch/(2*max(cw,ch)),device='cpu'),
        K_mhmr=get_camera_parameters(896,device='cpu'),img_mask=torch.tensor([True]),ray_mask=torch.tensor([False]),update=torch.tensor([True]),reset=torch.tensor([False]))


class FrozenFullFrontend:
    def __init__(self,repo,rich,dino_repo,workers=4):
        self.repo=Path(repo).resolve();self.rich=Path(rich).resolve();self.workers=workers
        sys.path[:0]=[str(self.repo),str(self.repo/'src')]
        os.environ['GUSH3R_DINO_HUB_LOCAL_REPO']=str(Path(dino_repo).resolve())
        from infer import load_model
        from src.dust3r.inference import inference_recurrent_lighter
        from src.dust3r.utils.camera import pose_encoding_to_camera
        self.inference=inference_recurrent_lighter;self.camera=pose_encoding_to_camera
        previous=Path.cwd()
        try:
            os.chdir(self.repo)
            self.model=load_model('cuda',SimpleNamespace(gs_conf_threshold=1.,bg_mask_threshold=.02,bg_mask_dilation=3,bg_voxel_size=.005,bg_gaussian_max=1,bg_cap_policy='random_legacy'))
        finally:os.chdir(previous)
        self.model.requires_grad_(False);self.model.eval();self.model.gs_mode='geometry_only';self.model.skip_inference_render_outputs=True
        self.mesh=self.model.human_gs_head.smplx_mesh
        for layer in self.mesh.layer.values():layer.requires_grad_(False);layer.eval()
        self.roi=np.array(json.loads((self.rich/'processed/supervision_v2/roi.json').read_text())['vertex_ids'])
        with np.load(self.rich/'processed/annotations_v1/smplx_topology.npz') as d:self.faces=d['faces']
        if not np.array_equal(self.faces,self.mesh.faces.cpu().numpy()):raise ValueError('SMPL-X topology mismatch')
        self.cameras=json.loads((self.rich/'processed/dataset_v1/cameras.json').read_text())['cameras']
        self.labels=LabelStore(self.rich,self.cameras);self.stash={}
        self.hook=self.model.backbone.register_forward_hook(lambda module,args,out:self.stash.update(dense=out.detach()))
        original=self.model._downstream_head
        def head(*args,**kwargs):
            if kwargs.get('smpl_token') is not None:self.stash['query']=kwargs['smpl_token'].detach()
            return original(*args,**kwargs)
        self.model._downstream_head=head
        compact=self.model._compact_recurrent_visible_humans
        def compact_visible(res,valid):
            result,indices=compact(res,valid)
            if 'query' in self.stash:self.stash['query']=self.stash['query'][:,indices]
            return result,indices
        self.model._compact_recurrent_visible_humans=compact_visible

    def prepare(self,clip):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda pair:prepare_frame(self.rich,pair[1],pair[0]),enumerate(clip['frames'])))

    @torch.no_grad()
    def features(self,frame,res,view):
        """Read only RGB/sensor metadata and predictions. No body/contact annotation access."""
        if 'smpl_rotmat' not in res or res['smpl_rotmat'].shape[1]==0:
            self.stash.clear();return [],np.empty((0,2)),[]
        c2w=self.camera(res['camera_pose'])[0]
        vertices,_,_=reconstruct_humangs_geometry(res,self.mesh,c2w)
        v=vertices.cpu().numpy();heads=res['smpl_loc'][0].cpu().numpy();ids=res['smpl_id'][0].cpu().numpy()
        query=self.stash['query'][0];dense=self.stash['dense'];self.stash.clear()
        if len(query)!=len(v) or len(heads)!=len(v):raise ValueError('Compacted person/query correspondence mismatch')
        side=int(round(dense.shape[1]**.5))
        dense=dense.reshape(1,side,side,1024).permute(0,3,1,2)
        global_rgb=F.adaptive_avg_pool2d(dense,4).permute(0,2,3,1).reshape(16,1024)
        gy,gx=torch.meshgrid(torch.linspace(-.75,.75,4,device='cuda'),torch.linspace(-.75,.75,4,device='cuda'),indexing='ij')
        global_rgb=torch.cat([global_rgb,torch.stack([gx,gy],-1).reshape(16,2)],-1)
        offset=torch.tensor([-4,-2,0,2,4],device='cuda',dtype=torch.float32)*14
        oy,ox=torch.meshgrid(offset,offset,indexing='ij');offset=torch.stack([ox,oy],-1).reshape(1,25,2)
        pointmap=res['pts3d_in_self_view'][0].detach().cpu().numpy()
        physical=self.cameras[frame['camera_key']]['Intrinsics'];_,kp=transformed_intrinsics(physical,*frame['raw_size_wh'])
        focal=float(view['K_mhmr'][0,0,0]);result=[];valid_people=[]
        for person in range(len(v)):
            sole=v[person,self.roi];center=sole.mean(1);normal=vertex_normals(v[person],self.faces)[self.roi]
            if (np.linalg.norm(normal,axis=-1)<.9).any():continue  # degenerate predicted foot mesh: skip this detection
            positions,point_normals,point_valid=scene_patch(pointmap,sole,center)
            head=res['smpl_transl'][0,person].detach().cpu().numpy()
            # Anchor projection to the RGB-detected head; compensate unknown global translation.
            uv=heads[person]+focal*(center[:,:2]/np.maximum(center[:,2:3],.1)-head[:2]/max(float(head[2]),.1))
            pixel=torch.from_numpy(uv).float().cuda()[:,None]+offset
            grid=(pixel+.5)*(2/896)-1
            rgb_valid=((grid>=-1)&(grid<=1)).all(-1)&torch.from_numpy(center[:,2]>.1).cuda()[:,None]
            local=F.grid_sample(dense.expand(2,-1,-1,-1),grid[:,None],mode='bilinear',padding_mode='zeros',align_corners=False).squeeze(2).transpose(1,2)
            local=torch.cat([local,grid],-1);local=local*rgb_valid[...,None]
            rgb=torch.cat([local,global_rgb[None].expand(2,-1,-1)],1).cpu().numpy()
            valid=torch.cat([rgb_valid,torch.ones((2,16),device='cuda',dtype=torch.bool)],1).cpu().numpy()
            pose=np.r_[res['smpl_rotmat'][0,person,:22].cpu().numpy().reshape(-1),res['smpl_shape'][0,person].cpu().numpy(),head]
            camera=np.r_[kp[0,0]/896,kp[1,1]/896,kp[0,2]/896,kp[1,2]/896,heads[person]/896]
            c=np.r_[pose,query[person].cpu().numpy(),camera]
            context=np.concatenate([np.repeat(c[None],2,axis=0),np.eye(2)],axis=1)
            item=dict(vertex_local=sole-center[:,None],vertex_normal=normal,point_local=positions,point_normal=point_normals,
                point_valid=point_valid,rgb_tokens=rgb,rgb_valid=valid,context=context)
            if any(not np.isfinite(x).all() for x in item.values()):continue  # skip non-finite frozen inputs instead of crashing
            result.append(item);valid_people.append(person)
        if not result:return [],np.empty((0,2),dtype=np.float32),[]
        return result,heads[valid_people],ids[valid_people].tolist()

    def extract(self,clip,views,seed):
        matched=[];failed=[];records=[]
        def callback(i,res,view):
            frame=clip['frames'][i]
            inputs,heads,ids=self.features(frame,res,view)
            boxes,targets=self.labels.targets(frame)
            assignment=match_targets(boxes,heads)
            for t,label,j in zip(frame['targets'],targets,assignment):
                record=dict(sample_id=t['sample_id'],frame_id=frame['frame_id'],subject=t['subject'],detected_people=len(heads),detection=j,
                            track_id=ids[j] if j>=0 else None,status='matched' if j>=0 else 'ambiguous_or_missing_target')
                records.append(record)
                if j>=0:matched.append((inputs[j],label))
                else:failed.append(label)
        # Preserve the trainable head's RNG across frozen extraction and exact replay.
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=True
            self.inference(views,self.model,'cuda',verbose=False,output_callback=callback,keep_outputs=False)
        arrays={}
        if matched:
            for key in INPUT_KEYS:arrays['input_'+key]=np.asarray([x[0][key] for x in matched],dtype=np.float16)
            for key in ['contact','contact_valid']:arrays['target_'+key]=np.asarray([x[1][key] for x in matched],dtype=np.uint8)
        for key in ['contact','contact_valid']:arrays['failed_'+key]=np.asarray([x[key] for x in failed],dtype=np.uint8).reshape(-1,2,64)
        self.stash.clear();gc.collect()
        return arrays,records

    def close(self):
        self.hook.remove();self.model._downstream_head=None;self.model._compact_recurrent_visible_humans=None
        del self.model;del self.mesh;self.stash.clear();gc.collect();torch.cuda.empty_cache()
