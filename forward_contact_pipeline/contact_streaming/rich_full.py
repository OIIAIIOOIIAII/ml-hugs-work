"""Full-data contact research: frozen inputs, separate matching labels, flexible head."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from .rich_assets import sha256
from .rich_supervision import pixel_transform,vertex_normals

INPUT_KEYS=('vertex_local','vertex_normal','point_local','point_normal','point_valid','rgb_tokens','rgb_valid','context')
CONTEXT_DIM=2011  # 198 rotations + 10 shape + 3 head + 1792 query + 6 camera/location + 2 region


def match_targets(boxes,heads,padding=.05):
    """Label association only: require mutually unique detection inside target box."""
    boxes=np.asarray(boxes).reshape(-1,2,2);heads=np.asarray(heads).reshape(-1,2)
    margin=(boxes[:,1]-boxes[:,0])*padding
    inside=((heads[None]>=boxes[:,None,0]-margin[:,None])&(heads[None]<=boxes[:,None,1]+margin[:,None])).all(-1)
    result=[]
    for row in inside:
        ids=np.flatnonzero(row)
        result.append(int(ids[0]) if len(ids)==1 and inside[:,ids[0]].sum()==1 else -1)
    return result


class LabelStore:
    """GT is read only after all-person frozen input features have been created."""
    def __init__(self,rich,cameras,cache_size=12):
        self.rich=Path(rich);self.cameras=cameras;self.cache_size=cache_size;self.cache=OrderedDict()

    def read(self,path,digest):
        key=str(path)
        if key not in self.cache:
            if sha256(path)!=digest:raise ValueError('Supervision source changed: '+key)
            with np.load(path,allow_pickle=False) as d:self.cache[key]={k:d[k] for k in d.files if k in ['vertices_multicam','frame_ids','contact','contact_valid']}
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def targets(self,frame):
        camera=self.cameras[frame['camera_key']];e=np.asarray(camera['CameraMatrix']);k=np.asarray(camera['Intrinsics'])
        a=np.asarray(pixel_transform(*frame['raw_size_wh'],512,896)['raw_to_pad'])
        boxes=[];labels=[]
        for target in frame['targets']:
            ann=target['annotation'];body=self.read(self.rich/ann['path'],ann['sha256']);row=ann['row']
            ref=target['target'];label=self.read(self.rich/'processed/supervision_v2'/ref['path'],ref['sha256'])
            if body['frame_ids'][row]!=frame['frame_id'] or label['frame_ids'][ref['row']]!=frame['frame_id']:raise ValueError('Supervision frame mismatch')
            xyz=body['vertices_multicam'][row]@e[:,:3].T+e[:,3]
            if (xyz[:,2]<=0).any():raise ValueError('Target behind camera')
            uv=xyz@k.T;uv=uv/uv[:,2:3];uv=uv@a.T
            boxes.append([uv[:,:2].min(0),uv[:,:2].max(0)])
            labels.append({key:label[key][ref['row']].astype(np.float32) for key in ['contact','contact_valid']})
        return np.asarray(boxes),labels


def scene_patch(points,vertices,center,neighbors=16,radius=.5,stride=4):
    """Predicted point-map neighborhoods; failed/remote geometry stays explicitly masked."""
    points=np.asarray(points)
    dx=np.roll(points,-1,axis=1)-np.roll(points,1,axis=1)
    dy=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    normals=np.cross(dx,dy);length=np.linalg.norm(normals,axis=-1,keepdims=True)
    normal=np.divide(normals,length,out=np.zeros_like(normals),where=length>1e-8)
    p=points[1:-1:stride,1:-1:stride].reshape(-1,3);n=normal[1:-1:stride,1:-1:stride].reshape(-1,3)
    keep=np.isfinite(p).all(1)&(p[:,2]>0)&(np.linalg.norm(n,axis=1)>.9)
    p=p[keep];n=n[keep]
    shape=vertices.shape[:-1]+(neighbors,3)
    if not len(p):return np.zeros(shape,np.float32),np.zeros(shape,np.float32),np.zeros(shape[:-1],np.float32)
    distance,idx=cKDTree(p).query(vertices.reshape(-1,3),k=min(neighbors,len(p)),workers=1)
    distance=distance.reshape(len(vertices.reshape(-1,3)),-1);idx=idx.reshape(distance.shape)
    if idx.shape[1]<neighbors:
        amount=neighbors-idx.shape[1];idx=np.pad(idx,((0,0),(0,amount)),mode='edge');distance=np.pad(distance,((0,0),(0,amount)),constant_values=np.inf)
    valid=(distance<=radius).reshape(shape[:-1]);local=p[idx].reshape(shape)-center[:,None,None,:]
    normals=n[idx].reshape(shape)
    return np.where(valid[...,None],local,0),np.where(valid[...,None],normals,0),valid.astype(np.float32)


class FullContactModel(nn.Module):
    """Shared vertex head with frozen local RGB, person context, sensor K and scene proxy."""
    required_inputs=INPUT_KEYS
    def __init__(self,hidden=128,heads=4,dropout=.1,rgb_dim=1026,context_dim=CONTEXT_DIM):
        super().__init__()
        if context_dim!=CONTEXT_DIM:raise ValueError('Version the context layout before changing its dimension')
        self.point=nn.Sequential(nn.Linear(7,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.vertex=nn.Sequential(nn.Linear(hidden+6,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.query=nn.Sequential(nn.LayerNorm(1792),nn.Linear(1792,hidden),nn.GELU())
        self.context=nn.Sequential(nn.Linear(219,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.semantic=nn.Parameter(torch.randn(2,64,hidden)*.02)
        self.rgb=nn.Linear(rgb_dim,hidden)
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True,dropout=dropout)
        self.norm=nn.LayerNorm(hidden)
        self.contact=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,inputs):
        c=inputs['context'];pose=c[:,:211].clone();pose[:,198:208]*=.3;pose[:,208:211]*=.1
        context=self.query(c[:,211:2003])+self.context(torch.cat([pose,c[:,2003:]],-1))
        region=c[:,-2:].argmax(-1)
        valid=inputs['point_valid'][...,None]
        point=self.point(torch.cat([inputs['point_local'],inputs['point_normal'],valid],-1))
        pooled=(point*valid).sum(-2)/valid.sum(-2).clamp_min(1)
        vertex=self.vertex(torch.cat([inputs['vertex_local'],inputs['vertex_normal'],pooled],-1))
        vertex=vertex+context[:,None]+self.semantic[region]
        rgb=self.rgb(inputs['rgb_tokens'])
        update,_=self.attention(vertex,rgb,rgb,key_padding_mask=~inputs['rgb_valid'].bool(),need_weights=False)
        return {'contact_logits':self.contact(self.norm(vertex+update)).squeeze(-1)}


def flatten_cache(arrays):
    """Only named input arrays are forwarded; target labels cannot leak through kwargs."""
    inputs={k:np.asarray(arrays['input_'+k],dtype=np.float32).reshape(-1,*arrays['input_'+k].shape[2:]) for k in INPUT_KEYS}
    targets={k:np.asarray(arrays['target_'+k],dtype=np.float32).reshape(-1,64) for k in ['contact','contact_valid']}
    return inputs,targets
