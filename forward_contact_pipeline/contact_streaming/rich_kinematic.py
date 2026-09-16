"""A shared joint regressor with explicit SMPL body-kinematic message passing."""
import torch
from torch import nn

BODY_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

class KinematicResidualNet(nn.Module):
    def __init__(self, dimension, mean, std, hidden=128, dropout=.1):
        super().__init__()
        if dimension != 2003:
            raise ValueError('Expected 211 pose/head + 1792 query dimensions')
        self.register_buffer('mean',torch.as_tensor(mean).float())
        self.register_buffer('std',torch.as_tensor(std).float().clamp_min(.01))
        self.register_buffer('output_scale',torch.tensor([.2]*66+[1.]*10+[.05]*3))
        adjacency=torch.eye(22)
        for joint,parent in enumerate(BODY_PARENTS):
            if parent>=0:adjacency[joint,parent]=adjacency[parent,joint]=1
        self.register_buffer('adjacency',adjacency/adjacency.sum(-1,keepdim=True))
        self.visual=nn.Sequential(nn.Linear(1792,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,64),nn.GELU())
        self.joint_identity=nn.Embedding(22,16)
        self.local=nn.Sequential(nn.Linear(9+13+64+16,64),nn.LayerNorm(64),nn.GELU())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64),nn.GELU()) for _ in range(3)])
        self.rotation=nn.Linear(64,3)
        self.global_head=nn.Sequential(nn.Linear(64+64+13,hidden),nn.GELU(),nn.Linear(hidden,13))
        for layer in [self.rotation,self.global_head[-1]]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,value):
        value=((value-self.mean)/self.std).clamp(-10,10)
        visual=self.visual(value[:,211:]);pose=value[:,:198].reshape(-1,22,9);global_pose=value[:,198:211]
        identity=self.joint_identity.weight[None].expand(len(value),-1,-1)
        h=self.local(torch.cat([pose,global_pose[:,None].expand(-1,22,-1),visual[:,None].expand(-1,22,-1),identity],-1))
        for message in self.messages:
            neighbours=torch.einsum('ij,bjd->bid',self.adjacency,h)
            h=h+message(torch.cat([h,neighbours],-1))
        out=torch.cat([self.rotation(h).reshape(-1,66),self.global_head(torch.cat([h.mean(1),visual,global_pose],-1))],-1)
        return out*self.output_scale
