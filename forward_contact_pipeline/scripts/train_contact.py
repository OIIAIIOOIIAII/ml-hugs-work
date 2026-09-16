#!/usr/bin/env python3
"""Train the frozen-backbone GRU/TCN contact correction network."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.data import ContactWindowDataset, feature_stats
from contact_streaming.losses import contact_loss
from contact_streaming.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["gru", "tcn"], default="gru")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-mode", choices=["all", "contact_only", "correction_contact"], default="all")
    return parser.parse_args()


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def normalize(batch, mean, std):
    batch["features"] = (batch["features"] - mean) / std
    return batch


def evaluate_loss(model, loader, device, mean, std, weights):
    model.eval(); totals = []
    with torch.no_grad():
        for batch in loader:
            batch = normalize(move(batch, device), mean, std)
            totals.append(float(contact_loss(model(batch["features"]), batch, weights)["total"].cpu()))
    return float(np.mean(totals))


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_data = ContactWindowDataset(args.data, "train", args.history)
    val_data = ContactWindowDataset(args.data, "val", args.history)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers)
    mean_np, std_np = feature_stats(train_data)
    device = torch.device(args.device)
    mean = torch.from_numpy(mean_np).to(device); std = torch.from_numpy(std_np).to(device)
    model = build_model(args.model, train_data.input_dim, args.hidden_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model, "input_dim": train_data.input_dim, "history": args.history,
        "hidden_dim": args.hidden_dim, "layers": args.layers, "dropout": args.dropout,
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed, "loss_mode": args.loss_mode,
    }
    if args.loss_mode == "all":
        weights = None
    elif args.loss_mode == "contact_only":
        weights = {"surface": 0.0, "velocity": 0.0, "penetration": 0.0, "pose": 0.0, "temporal": 0.0, "uncertainty": 0.0}
    else:
        # E1 mechanism setting: do not let auxiliary surface heads overwhelm
        # millimetre-scale correction residuals.
        weights = {"surface": 0.0, "velocity": 0.0, "penetration": 0.0, "pose": 5.0, "temporal": 0.0, "uncertainty": 0.0}
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for batch in train_loader:
            batch = normalize(move(batch, device), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                losses = contact_loss(model(batch["features"]), batch, weights)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(losses["total"].detach().cpu()))
        val_loss = evaluate_loss(model, val_loader, device, mean, std, weights)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        history.append(record); print(json.dumps(record), flush=True)
        checkpoint = {
            "config": config, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "feature_mean": mean_np, "feature_std": std_np, "epoch": epoch, "val_loss": val_loss,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if val_loss < best:
            best = val_loss; torch.save(checkpoint, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Best validation loss: {best:.6f}; checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
