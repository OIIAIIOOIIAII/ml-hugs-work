#!/usr/bin/env python3
"""Evaluate E1b with model-owned causal state; no teacher residual is read."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.anchorlock_rollout import rollout
from contact_streaming.metrics import binary_contact_metrics, foot_sliding, transition_f1
from contact_streaming.runtime import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test")); p.add_argument("--gate", type=float, default=0.7)
    p.add_argument("--apply-gate", choices=("soft", "hard"), default="soft", help="Action gate. soft matches rollout training; hard is an explicit deployment ablation.")
    p.add_argument("--min-surface-confidence", type=float, default=0.0, help="Abstain from applying/feeding back action below this local-geometry confidence.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model, ckpt = load_checkpoint(args.checkpoint, args.device); cfg = ckpt["config"]
    mean = torch.as_tensor(ckpt["feature_mean"], device=args.device); std = torch.as_tensor(ckpt["feature_std"], device=args.device)
    rows = []
    for path in sorted((args.data / args.split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as x: seq = {k: x[k].copy() for k in x.files}
        external_gate = torch.as_tensor((seq["surface_confidence"] >= args.min_surface_confidence).astype(np.float32), device=args.device)
        with torch.no_grad():
            prob, action = rollout(model, torch.as_tensor(seq["features"], device=args.device), torch.as_tensor(seq["surface_normal"], device=args.device),
                                   int(cfg["history"]), mean, std, float(cfg["max_action_m"]),
                                   hard_gate=args.gate if args.apply_gate == "hard" else None, external_gate=external_gate)
        prob, action = prob.cpu().numpy(), action.cpu().numpy()
        target = seq["contact"]; corrected = seq["anchor_position"] + action
        target_delta = seq["pose_residual"][:, :6].reshape(-1, 2, 3); clean = seq["anchor_position"] + target_delta
        row = binary_contact_metrics(prob, target); row.update({
            "transition_f1": transition_f1(prob, target),
            "correction_mae_m": float(np.mean(np.abs(corrected - clean))),
            "pre_foot_sliding_mps": foot_sliding(seq["anchor_position"], target, seq["timestamps"]),
            "post_foot_sliding_mps": foot_sliding(corrected, target, seq["timestamps"]),
            "abstention_rate": float((seq["surface_confidence"] < args.min_surface_confidence).mean()),
            "sequence": path.name,
        }); rows.append(row)
    numeric = [{k:v for k,v in row.items() if k != "sequence"} for row in rows]
    report = {"stage":"E2" if args.min_surface_confidence > 0 else "E1b","scope":"causal rollout; no teacher previous state","gate":args.gate,"apply_gate":args.apply_gate,"min_surface_confidence":args.min_surface_confidence,"num_sequences":len(rows),
              "macro_mean":{k:float(np.mean([r[k] for r in numeric])) for k in numeric[0]},"per_sequence":rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
