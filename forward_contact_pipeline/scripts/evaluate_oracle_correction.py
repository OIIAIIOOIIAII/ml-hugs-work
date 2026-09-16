#!/usr/bin/env python3
"""Sequence-safe E1 evaluation after applying low-dimensional corrections.

Uses only synthetic PROX drift arrays and their SDF-derived local geometry.
This is a mechanism evaluation, not a Human3R/GUSH3R result.  It reports
pre/post contact geometry, correction error and sequence-safe transition/slide
statistics rather than only raw network-head regression losses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.metrics import binary_contact_metrics, foot_sliding, transition_f1
from contact_streaming.runtime import load_checkpoint


def rule_prediction(seq: dict[str, np.ndarray], distance_m: float, speed_mps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = seq["features"][:, [18, 43]]
    speed = np.linalg.norm(seq["anchor_velocity"], axis=-1)
    contact = ((np.abs(observed) <= distance_m) & (speed <= speed_mps)).astype(np.float32)
    delta = -observed[..., None] * seq["surface_normal"] * contact[..., None]
    active = np.maximum(contact.sum(axis=1, keepdims=True), 1.0)
    root_xyz = delta.sum(axis=1) / active
    root = np.zeros((len(delta), 6), np.float32); root[:, :3] = root_xyz
    pose = np.zeros((len(delta), 12), np.float32); pose[:, :6] = (delta - root_xyz[:, None, :]).reshape(len(delta), 6)
    return contact, root, pose


def model_prediction(seq: dict[str, np.ndarray], checkpoint: str, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, payload = load_checkpoint(checkpoint, device)
    history = int(payload["config"]["history"])
    mean = torch.as_tensor(payload["feature_mean"], device=device)
    std = torch.as_tensor(payload["feature_std"], device=device)
    n = len(seq["features"])
    prob = np.zeros((n, 2), np.float32)
    root = np.zeros((n, 6), np.float32)
    pose = np.zeros((n, 12), np.float32)
    model.eval()
    with torch.no_grad():
        for end in range(history - 1, n):
            x = torch.as_tensor(seq["features"][end - history + 1:end + 1], device=device).unsqueeze(0)
            out = model((x - mean) / std)
            prob[end] = torch.sigmoid(out.contact_logits)[0].cpu().numpy()
            root[end] = out.root_residual[0].cpu().numpy()
            pose[end] = out.pose_residual[0].cpu().numpy()
    return prob, root, pose


def summarize_sequence(seq: dict[str, np.ndarray], probability: np.ndarray, root: np.ndarray, pose: np.ndarray, penetration_threshold_m: float, gate_threshold: float | None, max_correction_m: float, zero_root_on_apply: bool) -> dict[str, float]:
    target = seq["contact"].astype(np.float32)
    observed = seq["features"][:, [18, 43]]
    normals = seq["surface_normal"]
    pred_delta = pose[:, :6].reshape(-1, 2, 3)
    if not zero_root_on_apply:
        pred_delta = pred_delta + root[:, None, :3]
    if gate_threshold is not None:
        pred_delta = pred_delta * (probability >= gate_threshold)[..., None]
    if max_correction_m > 0:
        norm = np.linalg.norm(pred_delta, axis=-1, keepdims=True)
        pred_delta = np.where(norm > max_correction_m, pred_delta * (max_correction_m / np.maximum(norm, 1e-8)), pred_delta)
    target_delta = seq["root_residual"][:, None, :3] + seq["pose_residual"][:, :6].reshape(-1, 2, 3)
    post_distance = observed + np.sum(pred_delta * normals, axis=-1)
    clean_anchor = seq["anchor_position"] + target_delta
    corrected_anchor = seq["anchor_position"] + pred_delta
    active = target >= 0.5
    pre_depth = np.maximum(-observed, 0.0)
    post_depth = np.maximum(-post_distance, 0.0)
    contact_metrics = binary_contact_metrics(probability, target)
    return {
        **contact_metrics,
        "transition_f1": transition_f1(probability, target),
        "correction_mae_m": float(np.mean(np.abs(corrected_anchor - clean_anchor))),
        "root_correction_rmse_m": float(np.sqrt(np.mean((root[:, :3] - seq["root_residual"][:, :3]) ** 2))),
        # A strict <0 sign test is dominated by sub-millimetre SDF interpolation
        # noise after oracle correction.  The primary penetration ratio uses the
        # same 5 mm physical tolerance as the PROX label builder.
        "pre_penetration_ratio_gt_threshold": float((observed < -penetration_threshold_m).mean()),
        "post_penetration_ratio_gt_threshold": float((post_distance < -penetration_threshold_m).mean()),
        "pre_signed_negative_fraction": float((observed < 0).mean()),
        "post_signed_negative_fraction": float((post_distance < 0).mean()),
        "pre_penetration_depth_m": float(pre_depth.mean()),
        "post_penetration_depth_m": float(post_depth.mean()),
        "pre_contact_abs_distance_m": float(np.abs(observed[active]).mean()) if active.any() else 0.0,
        "post_contact_abs_distance_m": float(np.abs(post_distance[active]).mean()) if active.any() else 0.0,
        "pre_foot_sliding_mps": foot_sliding(seq["anchor_position"], target, seq.get("timestamps")),
        "post_foot_sliding_mps": foot_sliding(corrected_anchor, target, seq.get("timestamps")),
    }


def mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--mode", choices=("zero", "oracle", "rule", "model"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rule-distance-m", type=float, default=0.035)
    parser.add_argument("--rule-speed-mps", type=float, default=0.16)
    parser.add_argument("--penetration-threshold-m", type=float, default=0.005)
    parser.add_argument("--gate-threshold", type=float, default=None, help="Only apply residual to feet whose predicted contact probability reaches this value")
    parser.add_argument("--max-correction-m", type=float, default=0.08, help="Per-foot safety clamp; 0 disables it")
    parser.add_argument("--zero-root-on-apply", action="store_true", help="Execute only per-foot residuals; used by E1b where root target is intentionally fixed to zero")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "model" and not args.checkpoint:
        parser.error("--checkpoint is required for --mode model")
    paths = sorted((args.data / args.split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(args.data / args.split)
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            seq = {key: archive[key].copy() for key in archive.files}
        if args.mode == "zero":
            probability = np.zeros_like(seq["contact"], np.float32); root = np.zeros_like(seq["root_residual"]); pose = np.zeros_like(seq["pose_residual"])
        elif args.mode == "oracle":
            probability = seq["contact"].copy(); root = seq["root_residual"].copy(); pose = seq["pose_residual"].copy()
        elif args.mode == "rule":
            probability, root, pose = rule_prediction(seq, args.rule_distance_m, args.rule_speed_mps)
        else:
            probability, root, pose = model_prediction(seq, args.checkpoint, args.device)
        row = summarize_sequence(seq, probability, root, pose, args.penetration_threshold_m, args.gate_threshold, args.max_correction_m, args.zero_root_on_apply); row["sequence"] = path.name
        rows.append(row)
    numeric = [{k: v for k, v in row.items() if k != "sequence"} for row in rows]
    report = {
        "stage": "E1",
        "scope": "PROX synthetic-drift oracle-scene mechanism result only; no Human3R/GUSH3R claim",
        "mode": args.mode,
        "split": args.split,
        "penetration_threshold_m": args.penetration_threshold_m,
        "gate_threshold": args.gate_threshold,
        "max_correction_m": args.max_correction_m,
        "zero_root_on_apply": args.zero_root_on_apply,
        "num_sequences": len(rows),
        "macro_mean": mean_dict(numeric),
        "per_sequence": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
