#!/usr/bin/env python3
"""Train E1b foot-lock with model-owned causal state and multi-step rollout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.anchorlock_rollout import rollout
from contact_streaming.models import build_model


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-action-m", type=float, default=0.04)
    p.add_argument("--w-contact", type=float, default=1.0)
    p.add_argument("--w-residual", type=float, default=50.0)
    p.add_argument("--w-stick", type=float, default=2.0)
    p.add_argument("--w-release", type=float, default=2.0)
    p.add_argument("--w-slew", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pt when it exists.")
    return p.parse_args()


def load(root: Path, split: str) -> list[dict[str, np.ndarray]]:
    values = []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as x:
            values.append({key: x[key].copy() for key in x.files})
    if not values:
        raise FileNotFoundError(root / split)
    return values


def normalize_stats(seqs: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for seq in seqs:
        f = seq["features"].copy()
        for ci, ri in ((21, 22), (46, 47)):
            f[:, ci] = 0.0; f[:, ri : ri + 3] = 0.0
        values.append(f)
    mean = np.concatenate(values).mean(axis=0).astype(np.float32)
    std = np.concatenate(values).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def one_loss(model, seq, history, mean, std, max_action, weights, device):
    features = torch.as_tensor(seq["features"], device=device)
    normal = torch.as_tensor(seq["surface_normal"], device=device)
    contact = torch.as_tensor(seq["contact"], device=device)
    target = torch.as_tensor(seq["pose_residual"][:, :6].reshape(-1, 2, 3), device=device)
    position = torch.as_tensor(seq["anchor_position"], device=device)
    timestamps = torch.as_tensor(seq["timestamps"], device=device)
    probability, action = rollout(model, features, normal, history, mean, std, max_action)
    state = F.binary_cross_entropy(probability, contact)
    residual = F.smooth_l1_loss(action, target)
    corrected = position + action
    dt = torch.clamp(timestamps[1:] - timestamps[:-1], min=1e-4)
    speed = torch.linalg.norm(corrected[1:] - corrected[:-1], dim=-1) / dt[:, None]
    hold = contact[1:] * contact[:-1]
    stick = (speed * hold).sum() / torch.clamp(hold.sum(), min=1.0)
    release = (torch.linalg.norm(action, dim=-1) * (1.0 - contact)).mean()
    slew = torch.linalg.norm(action[1:] - action[:-1], dim=-1).mean()
    total = (weights["contact"] * state + weights["residual"] * residual + weights["stick"] * stick
             + weights["release"] * release + weights["slew"] * slew)
    return total, {"total": total, "state": state, "residual": residual, "stick": stick, "release": release, "slew": slew}


def evaluate(model, seqs, history, mean, std, max_action, weights, device):
    model.eval(); values = []
    with torch.no_grad():
        for seq in seqs:
            _, loss = one_loss(model, seq, history, mean, std, max_action, weights, device)
            values.append({k: float(v.detach().cpu()) for k, v in loss.items()})
    return {k: float(np.mean([x[k] for x in values])) for k in values[0]}


def main() -> None:
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train, val = load(args.data, "train"), load(args.data, "val")
    mean_np, std_np = normalize_stats(train)
    device = torch.device(args.device); mean = torch.as_tensor(mean_np, device=device); std = torch.as_tensor(std_np, device=device)
    model = build_model("gru", train[0]["features"].shape[-1], args.hidden_dim, 1, 0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {"contact": args.w_contact, "residual": args.w_residual, "stick": args.w_stick,
               "release": args.w_release, "slew": args.w_slew}
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"model": "gru", "input_dim": int(train[0]["features"].shape[-1]), "history": args.history,
              "hidden_dim": args.hidden_dim, "layers": 1, "dropout": 0.0, "loss_mode": "rollout_anchorlock",
              "max_action_m": args.max_action_m, "loss_weights": weights, "seed": args.seed}
    best = float("inf"); records = []; first_epoch = 1
    resume_path = args.output / "last.pt"
    history_path = args.output / "history.json"
    if args.resume and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("config") != config:
            raise ValueError(f"Refusing incompatible resume checkpoint: {resume_path}")
        model.load_state_dict(saved["model_state"])
        if "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])
        first_epoch = int(saved.get("epoch", 0)) + 1
        best = float(saved.get("best_val", saved.get("val_loss", float("inf"))))
        if history_path.exists():
            records = json.loads(history_path.read_text(encoding="utf-8"))
        print(json.dumps({"resume_from": str(resume_path), "first_epoch": first_epoch, "best_val": best}), flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train(); order = list(range(len(train))); random.Random(args.seed + epoch).shuffle(order); losses = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            total, _ = one_loss(model, train[index], args.history, mean, std, args.max_action_m, weights, device)
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(total.detach().cpu()))
        val_loss = evaluate(model, val, args.history, mean, std, args.max_action_m, weights, device)
        record = {"epoch": epoch, "train_total": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_loss.items()}}
        records.append(record); print(json.dumps(record), flush=True)
        improved = val_loss["total"] < best
        if improved:
            best = val_loss["total"]
        checkpoint = {"config": config, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "feature_mean": mean_np, "feature_std": std_np, "epoch": epoch, "val_loss": val_loss["total"], "best_val": best}
        torch.save(checkpoint, args.output / "last.pt")
        if improved:
            torch.save(checkpoint, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
