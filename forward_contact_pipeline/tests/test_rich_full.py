import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from contact_streaming.rich_full import FullContactModel,INPUT_KEYS,flatten_cache,match_targets,scene_patch
from train_rich_full_contact import train_shard


def fixture():
    rng=np.random.default_rng(17);shape=(2,2)
    arrays={}
    for k,tail in [('vertex_local',(64,3)),('vertex_normal',(64,3)),('point_local',(64,16,3)),('point_normal',(64,16,3)),('rgb_tokens',(41,1026)),('context',(2011,))]:
        arrays['input_'+k]=(rng.normal(size=shape+tail)*.1).astype(np.float16)
    arrays['input_context'][:,:,-2:]=np.eye(2)
    arrays['input_point_valid']=np.zeros(shape+(64,16),np.float16)
    arrays['input_rgb_valid']=np.ones(shape+(41,),np.float16)
    arrays['target_contact']=rng.integers(0,2,shape+(64,),dtype=np.uint8)
    arrays['target_contact_valid']=np.ones(shape+(64,),np.uint8)
    return arrays


class FullTrainingContracts(unittest.TestCase):
    def test_multipeople_label_matching_and_ambiguity(self):
        box=np.array([[[0,0],[10,20]],[[30,0],[40,20]]])
        self.assertEqual(match_targets(box,[[5,2],[35,2],[80,90]]),[0,1])
        self.assertEqual(match_targets(box,[[5,2],[6,3],[35,2]]),[-1,2])
        self.assertEqual(match_targets(np.repeat(box[:1],2,axis=0),[[5,2]]),[-1,-1])
        self.assertEqual(match_targets(box,np.empty((0,2))),[-1,-1])

    def test_labels_never_enter_features_and_sensor_input_reaches_output(self):
        torch.manual_seed(8);a=fixture();x,_=flatten_cache(a)
        model=FullContactModel(hidden=32,dropout=0).eval();x={k:torch.from_numpy(v) for k,v in x.items()}
        original=model(x)['contact_logits']
        a['target_contact'][:]=0;after,_=flatten_cache(a)
        for k in INPUT_KEYS:np.testing.assert_array_equal(after[k],x[k].numpy())
        changed={k:v.clone() for k,v in x.items()};changed['context'][:,2003:2005]+=.5
        self.assertGreater(float((model(changed)['contact_logits']-original).abs().max()),1e-6)
        x['target_contact']=torch.full_like(original,float('nan'))
        torch.testing.assert_close(model(x)['contact_logits'],original,rtol=0,atol=0)

    def test_optimizer_and_rng_resume_matches_uninterrupted_updates(self):
        torch.manual_seed(9);a=fixture();model=FullContactModel(hidden=32,dropout=.1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        train_shard(model,optimizer,a,7,2,'cpu')
        saved_model=copy.deepcopy(model.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state()
        train_shard(model,optimizer,a,19,2,'cpu');expected=copy.deepcopy(model.state_dict())
        resumed=FullContactModel(hidden=32,dropout=.1);resumed.load_state_dict(saved_model)
        opt=torch.optim.AdamW(resumed.parameters(),lr=.001);opt.load_state_dict(saved_optimizer);torch.set_rng_state(rng)
        train_shard(resumed,opt,a,19,2,'cpu')
        for k,v in resumed.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)

    def test_distant_scene_proxy_remains_explicitly_masked(self):
        yy,xx=np.mgrid[:12,:12];p=np.stack([xx*.01,yy*.01,np.ones_like(xx)],-1).astype(np.float32)
        vertices=np.zeros((2,64,3),np.float32);center=np.zeros((2,3),np.float32)
        local,normals,valid=scene_patch(p,vertices,center,stride=1)
        self.assertEqual(float(valid.sum()),0);self.assertEqual(float(abs(local).sum()+abs(normals).sum()),0)
        vertices[...,2]=1;center[:,2]=1
        local,normals,valid=scene_patch(p,vertices,center,stride=1)
        self.assertTrue(valid.any());np.testing.assert_allclose(np.linalg.norm(normals[valid.astype(bool)],axis=-1),1,atol=1e-6)


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
