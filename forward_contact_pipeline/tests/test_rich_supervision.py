import sys
import json
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_supervision import pixel_transform, project, frustum, transform_uv, vertex_normals, prepare
from contact_streaming.rich_assets import save_json, sha256

class SupervisionTests(unittest.TestCase):
    def test_resize_crop_portrait_and_landscape(self):
        for wh,resized,crop in [((4112,3008),(512,375),(512,368)),((3008,4112),(375,512),(368,512)),
                               ((4096,2700),(512,338),(512,336)),((512,512),(512,512),(512,384))]:
            with self.subTest(wh=wh):
                t=pixel_transform(*wh,pad_to=896)
                self.assertEqual(tuple(t['resized_size_wh']),resized)
                self.assertEqual(tuple(t['crop_size_wh']),crop)
                np.testing.assert_allclose(t['raw_to_pad'],np.array(t['crop_to_pad'])@np.array(t['raw_to_crop']))
                a=np.array(t['raw_to_crop']); corners=np.array([[0,0,1],[crop[0]-1,crop[1]-1,1.]])
                original=corners@np.linalg.inv(a).T
                np.testing.assert_allclose(transform_uv(original[:,:2],a),corners[:,:2],atol=1e-12)

    def test_behind_camera_and_crop_exclusion(self):
        cam={'CameraMatrix':np.eye(4)[:3].tolist(),'Intrinsics':np.eye(3).tolist()}
        points=np.array([[1,1,1],[1,1,-1],[7,1,1],[0,0,0],[np.nan,0,1]])
        uv,z=project(points,cam)
        self.assertEqual(frustum(uv,z,5,5).tolist(),[True,False,False,False,False])
        moved=transform_uv(uv,[[1,0,-2],[0,1,0],[0,0,1]])
        self.assertFalse(frustum(moved,z,5,5)[0])

    def test_normals_orientation_and_degenerate_faces(self):
        v=np.array([[0.,0,0],[1,0,0],[0,0,1],[2,0,0]])
        n=vertex_normals(v,np.array([[0,1,2],[0,1,3]]))
        np.testing.assert_allclose(n[:3],[[0,-1,0]]*3)
        np.testing.assert_array_equal(n[3],[0,0,0])


def dataset_fixture(root):
    data=root/'processed/dataset_v1'; data.mkdir(parents=True)
    cams={}
    for split,seq in [('train','A'),('val','B')]:
        path=root/f'processed/annotations_v1/shards/{seq}/000/000000.npz'
        path.parent.mkdir(parents=True)
        v=np.zeros((2,10475,3),np.float32); v[...,2]=1
        valid=np.ones((2,10475),bool); valid[1]=False
        labels=np.zeros((2,10475),bool); labels[:,1]=True
        np.savez(path,frame_ids=[5,7],vertices_multicam=v,contact_smplx=labels,
                 contact_smplx_valid=valid,s2m_dist_id=np.zeros_like(v))
        rows=[]
        for cam in [0,1]:
            key=f'{seq}/scan/cam_{cam:02}'
            e=np.eye(4)[:3]; e[0,3]=cam*1000
            cams[key]=dict(status='structurally_consistent',raw_width=100,raw_height=80,
                           CameraMatrix=e.tolist(),Intrinsics=[[30,0,50],[0,30,40],[0,0,1]])
            for row,frame in enumerate([5,7]):
                rows.append(dict(sample_id=f'{seq}/000/cam_{cam:02}/{frame:05}',split=split,
                    sequence=seq,subject='000',camera_id=cam,camera_key=key,frame_id=frame,exif_orientation=1,
                    annotation=dict(path=str(path.relative_to(root)),sha256=sha256(path),row=row)))
        (data/f'samples_{split}.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
    save_json(data/'cameras.json',dict(cameras=cams))
    save_json(data/'splits.json',{})
    (data/'image_headers.jsonl').write_text('')
    return np.array([[0,1],[2,3]]),dict(name='test_fixed_roi')

class FullPreparationTests(unittest.TestCase):
    def test_invalid_labels_offscreen_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); roi=dataset_fixture(root); out=root/'output'
            with patch('contact_streaming.rich_supervision.sole_roi',return_value=roi), redirect_stdout(io.StringIO()):
                first=prepare(root,out,root/'template',workers=2)
                self.assertEqual(first['counts']['train']['eligible'],1)
                self.assertEqual(first['counts']['val']['missing_smplx_labels'],2)
                self.assertEqual(first['counts']['train']['soles_outside_crop'],2)
                self.assertFalse(first['training_ready'])
                before=(out/'targets/A/000/000000.npz').stat().st_mtime_ns
                resumed=prepare(root,out,root/'template',workers=2)
                self.assertEqual(first['counts'],resumed['counts'])
                self.assertEqual(before,(out/'targets/A/000/000000.npz').stat().st_mtime_ns)
                with (out/'frustum/A/000/000000.npz').open('ab') as f: f.write(b'corruption')
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    prepare(root,out,root/'template',workers=2)

    def test_source_hash_and_frame_association_fail_closed(self):
        for defect in ['hash','frame']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); roi=dataset_fixture(root)
                if defect=='hash':
                    path=root/'processed/annotations_v1/shards/A/000/000000.npz'
                    with path.open('ab') as f: f.write(b'changed')
                else:
                    path=root/'processed/dataset_v1/samples_train.jsonl'
                    records=[json.loads(l) for l in path.read_text().splitlines()]
                    records[0]['frame_id']=6
                    path.write_text(''.join(json.dumps(s)+'\n' for s in records))
                with patch('contact_streaming.rich_supervision.sole_roi',return_value=roi), redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError): prepare(root,root/'out',root/'template',workers=2)
                self.assertFalse((root/'out/report.json').exists())

if __name__=='__main__': unittest.main()
