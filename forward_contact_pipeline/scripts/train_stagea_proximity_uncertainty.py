#!/usr/bin/env python3
"""Calibrate Stage-A continuous-proximity uncertainty under proxy corruption.

This is a geometry-only oracle experiment.  The reliability head represents a
Laplace scale for the signed-SDF proximity residual; low predicted scale means
the online controller may use the relation token, high scale means abstain.
It never treats PROX's ROI-level binary label as dense vertex contact.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.stagea import LocalRelationEncoder


class Relations(Dataset):
    def __init__(self, manifests):
        self.sequences, self.index = [], []
        for manifest_path in manifests:
            file = Path(manifest_path); manifest = json.loads(file.read_text())
            with np.load(file.parent / manifest['payload'], allow_pickle=False) as d:
                valid = d['scene_point_valid'] if 'scene_point_valid' in d else np.ones(d['scene_point_local'].shape[:-1], np.float32)
                arrays = tuple(np.asarray(x, np.float32) for x in (d['roi_vertex_local'], d['roi_vertex_normal_local'], d['scene_point_local'], d['scene_normal_local'], valid, d['vertex_proximity']))
            sequence = len(self.sequences); self.sequences.append(arrays)
            self.index.extend((sequence, frame, foot) for frame in range(arrays[0].shape[0]) for foot in range(2))
    def __len__(self): return len(self.index)
    def __getitem__(self, index):
        sequence, frame, foot = self.index[index]; a = self.sequences[sequence]
        return tuple(torch.from_numpy(x[frame, foot]).float() for x in a)


def uncertainty(network, batch):
    output = network(*batch[:5])
    # Differentiably bounded Laplace scale in metres.  A hard post-exp clamp
    # makes the reliability head saturate at the maximum and kills its
    # gradient, so parameterize log-scale through a sigmoid instead.
    log_min, log_max = float(np.log(1e-4)), float(np.log(5e-2))
    log_scale = log_min + (log_max - log_min) * torch.sigmoid(output['reliability_logits'])
    scale = torch.exp(log_scale)
    return output['proximity'], scale


def metrics(network, loader, device):
    network.eval(); error, scale = [], []
    with torch.no_grad():
        for raw in loader:
            batch = [x.to(device) for x in raw]
            proximity, predicted_scale = uncertainty(network, batch)
            error.append((proximity - batch[5]).abs().flatten().cpu())
            scale.append(predicted_scale.flatten().cpu())
    error, scale = torch.cat(error), torch.cat(scale)
    order = torch.argsort(scale)  # retain the most reliable elements first
    curve = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(len(error) * coverage))
        curve[str(coverage)] = float(error[order[:count]].mean())
    bins = []
    for ids in torch.tensor_split(order, 5):
        bins.append({'count': int(len(ids)), 'mean_predicted_scale_m': float(scale[ids].mean()), 'mean_absolute_error_m': float(error[ids].mean())})
    return {'mae_m': float(error.mean()), 'rmse_m': float(torch.sqrt((error.square()).mean())), 'mean_scale_m': float(scale.mean()), 'laplace_ratio_abs_error_over_scale': float((error / scale).mean()), 'coverage_risk_mae_m': curve, 'calibration_bins_low_to_high_scale': bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40); parser.add_argument('--batch', type=int, default=32); parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    split = json.loads(args.split.read_text()); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = {name: Relations(paths) for name, paths in split['splits'].items()}
    loaders = {name: DataLoader(data, batch_size=args.batch, shuffle=name == 'train') for name, data in datasets.items()}
    network = LocalRelationEncoder().to(device); optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-4)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        network.train(); losses = []
        for raw in loaders['train']:
            batch = [x.to(device) for x in raw]; proximity, scale = uncertainty(network, batch)
            # Laplace NLL teaches a predictive error scale without access to a
            # corruption-family ID at inference.
            residual = (proximity - batch[5]).abs()
            loss = (residual / scale + torch.log(scale)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach()))
        validation = metrics(network, loaders['val'], device)
        record = {'epoch': epoch, 'train_laplace_nll': float(np.mean(losses)), **validation}; history.append(record)
        if best is None or validation['mae_m'] < best['mae_m']:
            best = {**record, 'state': {k: v.detach().cpu() for k, v in network.state_dict().items()}}
    network.load_state_dict(best.pop('state'))
    output = {'kind': 'stagea_proximity_uncertainty_proxy_ood', 'scope': 'oracle local relation; no RGB/Gaussian/controller', 'split': str(args.split), 'selection': 'minimum validation proximity MAE', 'samples': {k: len(v) for k, v in datasets.items()}, 'best_validation': best, 'test': metrics(network, loaders['test'], device), 'history': history}
    args.output.mkdir(parents=True); torch.save(network.state_dict(), args.output / 'best.pt'); (args.output / 'metrics.json').write_text(json.dumps(output, indent=2)); print(json.dumps({'best_validation': best, 'test': output['test']}, ensure_ascii=False))

if __name__ == '__main__': main()
