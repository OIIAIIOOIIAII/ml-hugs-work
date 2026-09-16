"""Versioned RICH sole targets and full frustum audit (no pixel rendering).

GT arrays in this module are supervision/evaluation only. Frustum masks do not
claim visibility through scene occlusion or self occlusion.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from .rich_assets import save_json, sha256
from .rich_dataset import read_json, write_line


def pixel_transform(width, height, size=512, pad_to=None):
    """GUSH3R load_images/pad_image, integer pixel centers, identity EXIF.

    Pixel-center resize x'=(x+.5)*scale-.5; crop/pad are integer shifts.
    Intrinsics here are evaluation intrinsics, never injected into the model.
    """
    edge = round(size * max(width / height, height / width)) if size == 224 else size
    rw, rh = (round(v * edge / max(width, height)) for v in (width, height))
    cx, cy = rw // 2, rh // 2
    if size == 224:
        hw = hh = min(cx, cy)
    else:
        hw, hh = (2 * cx // 16) * 8, (2 * cy // 16) * 8
        if rw == rh:
            hh = 3 * hw / 4
    left, top, right, bottom = map(round, (cx-hw, cy-hh, cx+hw, cy+hh))
    cw, ch = right-left, bottom-top
    sx, sy = rw/width, rh/height
    a = np.array([[sx, 0, (sx-1)/2-left], [0, sy, (sy-1)/2-top], [0, 0, 1.]])
    result = dict(raw_size_wh=[width,height], resized_size_wh=[rw,rh],
                  crop_xyxy=[left,top,right,bottom], crop_size_wh=[cw,ch],
                  raw_to_crop=a.tolist(), pixel_convention='integer centers; half-pixel resize')
    if pad_to is not None:
        scale = min(pad_to/ch, pad_to/cw)
        pw, ph = int(cw*scale), int(ch*scale)
        pl, pt = (pad_to-pw)//2, (pad_to-ph)//2
        sx, sy = pw/cw, ph/ch
        b = np.array([[sx,0,(sx-1)/2+pl],[0,sy,(sy-1)/2+pt],[0,0,1.]])
        result.update(padded_size_wh=[pad_to,pad_to], crop_to_pad=b.tolist(), raw_to_pad=(b@a).tolist())
    return result


def vertex_normals(vertices, faces):
    tri = vertices[faces]
    area = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for i in range(3):
        np.add.at(normals, faces[:,i], area)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length>1e-12)


def sole_roi(template, topology, count=64):
    with np.load(template, allow_pickle=False) as d:
        v, f, w = d['v_template'], d['f'], d['weights']
    with np.load(topology, allow_pickle=False) as d:
        if not np.array_equal(f, d['faces']):
            raise ValueError('Template and annotation topology differ')
    if v.shape != (10475,3) or w.shape != (10475,55):
        raise ValueError('Expected SMPL-X topology and joints')
    normals = vertex_normals(v,f)
    selected, candidates = [], []
    # Verified against smplx.joint_names.JOINT_NAMES: ankle 7/8, foot 10/11.
    for joints in ([7,10],[8,11]):
        foot = np.flatnonzero(np.isin(w.argmax(axis=1), joints))
        sole = foot[(normals[foot,1]<-.25) & (v[foot,1]<=np.quantile(v[foot,1],.55))]
        if len(sole)<count:
            raise ValueError('Insufficient neutral-template sole vertices')
        chosen = [0]  # smallest vertex ID; all ties use smallest candidate index
        distances = np.full(len(sole),np.inf)
        for _ in range(count-1):
            distances = np.minimum(distances, np.sum((v[sole]-v[sole[chosen[-1]]])**2,axis=-1))
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
        selected.append(sole[chosen]); candidates.append(len(sole))
    ids = np.array(selected,dtype=np.int64)
    return ids, dict(schema_version=1, name='smplx_neutral_sole_fps64_v1',
        vertex_ids=ids.tolist(), sides=['left','right'], candidate_counts=candidates,
        dominant_joints=[[7,10],[8,11]], joint_names=[['left_ankle','left_foot'],['right_ankle','right_foot']],
        selection='neutral template only: normal_y < -0.25 and y <= foot quantile .55; deterministic FPS64',
        template_sha256=sha256(template), topology_sha256=sha256(topology),
        label_values_used_for_selection=False)


def project(vertices, camera):
    e, k = np.asarray(camera['CameraMatrix']), np.asarray(camera['Intrinsics'])
    p = vertices @ e[:,:3].T + e[:,3]
    q = p @ k.T
    z = p[...,2]
    uv = np.divide(q[...,:2],q[...,2:],out=np.zeros_like(q[...,:2]),where=np.abs(q[...,2:])>1e-12)
    return uv,z


def frustum(uv,z,width,height):
    return ((z>1e-6) & np.isfinite(uv).all(axis=-1) & (uv[...,0]>=0) & (uv[...,0]<width)
            & (uv[...,1]>=0) & (uv[...,1]<height))


def transform_uv(uv,a):
    a = np.asarray(a)
    return uv @ a[:2,:2].T + a[:2,2]


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays)
    tmp.replace(path)


def prepare(root, output, template, workers=4, size=512):
    root, output, template = Path(root).resolve(), Path(output).resolve(), Path(template).resolve()
    output.mkdir(parents=True,exist_ok=True)
    dataset = root/'processed/dataset_v1'
    ids, roi = sole_roi(template,root/'processed/annotations_v1/smplx_topology.npz')
    sources = {name:sha256(dataset/name) for name in ['samples_train.jsonl','samples_val.jsonl','cameras.json','splits.json','image_headers.jsonl']}
    config = dict(schema_version=1, kind='GT_supervision_and_frustum_only', sources=sources,
        roi=roi, frontend_image_size=size, implementation_sha256=sha256(Path(__file__)),
        masking='contact_valid AND crop_frustum; no occlusion claim', training_ready=False)
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (output/'config.json').exists() and read_json(output/'config.json') != config:
        raise ValueError('Output configuration/source changed; use a new version directory')
    save_json(output/'config.json',config); save_json(output/'roi.json',roi)
    cameras = read_json(dataset/'cameras.json')['cameras']
    transforms = {}
    for key, cam in cameras.items():
        if cam['status']!='structurally_consistent':
            continue
        t = pixel_transform(cam['raw_width'],cam['raw_height'],size)
        t['K_crop_evaluation_only']=(np.asarray(t['raw_to_crop'])@np.asarray(cam['Intrinsics'])).tolist()
        transforms[key]=t
    save_json(output/'image_transforms.json',transforms)
    groups = defaultdict(list)
    for split in ['train','val']:
        with (dataset/f'samples_{split}.jsonl').open() as f:
            for line in f:
                sample=json.loads(line); groups[sample['annotation']['path']].append(sample)
    total=len(groups); started=time.time()

    def process(item):
        source, samples = item
        samples=sorted(samples,key=lambda s:(s['camera_id'],s['frame_id']))
        expected={s['annotation']['sha256'] for s in samples}
        payload=(root/source).read_bytes()
        source_hash=hashlib.sha256(payload).hexdigest()
        if expected!={source_hash}:
            raise ValueError('Source shard checksum mismatch: '+source)
        rel=Path(source).relative_to('processed/annotations_v1/shards')
        target=output/'targets'/rel; visibility=output/'frustum'/rel
        receipt=output/'receipts'/rel.with_suffix('.json')
        sample_hash=hashlib.sha256(json.dumps(samples,sort_keys=True).encode()).hexdigest()
        if receipt.exists():
            old=read_json(receipt)
            if old['signature']==signature and old['samples_sha256']==sample_hash and old['source_sha256']==source_hash:
                if sha256(target)==old['target_sha256'] and sha256(visibility)==old['frustum_sha256']:
                    return samples,old
                raise ValueError('Existing output checksum mismatch: '+str(receipt))
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            frames=data['frame_ids']; vertices=data['vertices_multicam'].astype(np.float64)
            labels=data['contact_smplx'][:,ids].astype(np.float32)
            valid=data['contact_smplx_valid'][:,ids].astype(bool)
            delta=data['s2m_dist_id'][:,ids]
        if not np.isfinite(vertices).all() or not np.isfinite(labels).all() or not np.isin(labels,[0,1]).all():
            raise ValueError('Nonfinite geometry or nonbinary labels')
        distance=np.linalg.norm(delta,axis=-1)
        distance_valid=np.isfinite(delta).all(axis=-1)
        atomic_npz(target,frame_ids=frames,vertex_ids=ids,contact=labels,contact_valid=valid,
                   unsigned_surface_distance=np.where(distance_valid,distance,0).astype(np.float32),
                   unsigned_surface_distance_valid=distance_valid)
        n=len(samples); raw_mask=np.zeros((n,2,len(ids[0])),bool); crop_mask=np.zeros_like(raw_mask)
        raw_counts=np.zeros(n,np.int32); crop_counts=np.zeros(n,np.int32); positive_depth=np.zeros(n,np.int32)
        projected=np.zeros((n,2,len(ids[0]),2),np.float32)
        by_camera=defaultdict(list)
        for i,s in enumerate(samples): by_camera[s['camera_key']].append((i,s))
        for key, indexed in by_camera.items():
            cam=cameras[key]; tr=transforms[key]
            uv,z=project(vertices,cam)
            raw=frustum(uv,z,cam['raw_width'],cam['raw_height'])
            cropped=transform_uv(uv,tr['raw_to_crop'])
            crop=frustum(cropped,z,*tr['crop_size_wh'])
            for i,s in indexed:
                row=s['annotation']['row']
                if int(frames[row])!=s['frame_id'] or s['exif_orientation']!=1:
                    raise ValueError('Frame/EXIF association mismatch')
                raw_mask[i]=raw[row,ids]; crop_mask[i]=crop[row,ids]
                raw_counts[i]=raw[row].sum(); crop_counts[i]=crop[row].sum()
                positive_depth[i]=(z[row]>1e-6).sum(); projected[i]=cropped[row,ids]
        rows=np.array([s['annotation']['row'] for s in samples],np.int64)
        supervision_mask=valid[rows] & crop_mask
        atomic_npz(visibility,sample_ids=np.array([s['sample_id'] for s in samples]),
            frame_ids=np.array([s['frame_id'] for s in samples]),annotation_rows=rows,
            sole_raw_frustum=raw_mask,sole_crop_frustum=crop_mask,sole_crop_uv=projected,
            contact_supervision_mask=supervision_mask,body_raw_inbounds=raw_counts,
            body_crop_inbounds=crop_counts,body_positive_depth=positive_depth)
        # Immediate shape/mask readback and summary; completion independently audits again.
        with np.load(visibility,allow_pickle=False) as d:
            if not np.array_equal(d['contact_supervision_mask'],supervision_mask): raise ValueError('Write readback failed')
        summary=dict(signature=signature,source_sha256=source_hash,samples_sha256=sample_hash,
            source=source,target=str(target.relative_to(output)),frustum=str(visibility.relative_to(output)),
            target_sha256=sha256(target),frustum_sha256=sha256(visibility),
            frames=len(frames),samples=n,split=samples[0]['split'],
            label_valid=int(valid.sum()),label_positive=int((labels*valid).sum()),
            view_valid=int(supervision_mask.sum()),view_positive=int((labels[rows]*supervision_mask).sum()))
        save_json(receipt,summary)
        return samples,summary

    counts={s:Counter() for s in ['train','val']}; artifacts=[]
    with ExitStack() as stack:
        streams={s:stack.enter_context((output/f'samples_{s}.jsonl.partial').open('w')) for s in counts}
        excluded=stack.enter_context((output/'excluded.jsonl.partial').open('w'))
        pool=stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        for done,(samples,receipt) in enumerate(pool.map(process,sorted(groups.items())),1):
            artifacts.append(receipt)
            split=receipt['split']; tally=counts[split]
            for key in ['frames','samples','label_valid','label_positive','view_valid','view_positive']: tally[key]+=receipt[key]
            with np.load(output/receipt['target'],allow_pickle=False) as tf, np.load(output/receipt['frustum'],allow_pickle=False) as vf:
                t={k:tf[k] for k in ['contact_valid']}
                v={k:vf[k] for k in ['body_crop_inbounds','sole_crop_frustum','body_raw_inbounds']}
                for i,s in enumerate(samples):
                    row=s['annotation']['row']; reasons=[]
                    if not t['contact_valid'][row].any(): reasons.append('missing_smplx_labels')
                    if v['body_crop_inbounds'][i]==0: reasons.append('body_outside_crop')
                    if not v['sole_crop_frustum'][i].any(): reasons.append('soles_outside_crop')
                    if reasons:
                        tally['excluded']+=1
                        for reason in reasons: tally[reason]+=1
                        write_line(excluded,dict(sample_id=s['sample_id'],split=split,reasons=reasons))
                    else:
                        tally['eligible']+=1
                        s.update(target={'path':receipt['target'],'sha256':receipt['target_sha256'],'row':row},
                            frustum={'path':receipt['frustum'],'sha256':receipt['frustum_sha256'],'row':i},
                            status='supervision_eligible_frontend_pending')
                        write_line(streams[split],s)
                    if v['body_raw_inbounds'][i]==0: tally['body_outside_raw']+=1
                    if v['body_crop_inbounds'][i]<10475//2: tally['body_less_than_half_in_crop']+=1
            if done%20==0 or done==total:
                progress=dict(status='processing',shards=done,total_shards=total,elapsed_s=time.time()-started,counts=counts)
                save_json(output/'progress.json',progress); print(json.dumps(progress),flush=True)
    for name in ['samples_train.jsonl','samples_val.jsonl','excluded.jsonl']:
        (output/(name+'.partial')).replace(output/name)
    save_json(output/'artifacts.json',artifacts)
    report=dict(schema_version=1,status='complete',supervision_ready=True,training_ready=False,
        counts=counts,shards=total,elapsed_s=time.time()-started,roi=roi['name'],
        limitations=['frustum is not occlusion visibility','real frozen frontend and geometry audit required',
                      'unsigned s2m norm is a separate optional target; not signed proximity',
                      'internal split of official train; official val/test not included'],
        hashes={name:sha256(output/name) for name in ['artifacts.json','roi.json','image_transforms.json','samples_train.jsonl','samples_val.jsonl','excluded.jsonl']})
    save_json(output/'report.json',report); save_json(output/'progress.json',report)
    return report
