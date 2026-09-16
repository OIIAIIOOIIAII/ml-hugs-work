#!/usr/bin/env python3
"""Evaluate a trained contact model on validation or test sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.data import ContactWindowDataset
from contact_streaming.losses import contact_loss
from contact_streaming.metrics import geometric_metrics
from contact_streaming.runtime import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = checkpoint["config"]
    dataset = ContactWindowDataset(args.data, args.split, config["history"])
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    mean = torch.as_tensor(checkpoint["feature_mean"], device=args.device)
    std = torch.as_tensor(checkpoint["feature_std"], device=args.device)
    collected = {key: [] for key in ["probability", "contact", "distance_pred", "distance_gt", "root_pred", "root_gt"]}
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            batch["features"] = (batch["features"] - mean) / std
            output = model(batch["features"])
            weights = None if config.get("loss_mode", "all") == "all" else {"surface": 0.0, "velocity": 0.0, "penetration": 0.0, "pose": 0.0, "temporal": 0.0, "uncertainty": 0.0}
            losses.append(float(contact_loss(output, batch, weights)["total"].cpu()))
            collected["probability"].append(torch.sigmoid(output.contact_logits).cpu().numpy())
            collected["contact"].append(batch["contact"].cpu().numpy())
            collected["distance_pred"].append(output.surface_distance.cpu().numpy())
            collected["distance_gt"].append(batch["surface_distance"].cpu().numpy())
            collected["root_pred"].append(output.root_residual.cpu().numpy())
            collected["root_gt"].append(batch["root_residual"].cpu().numpy())
    values = {key: np.concatenate(items) for key, items in collected.items()}
    metrics = geometric_metrics(
        values["probability"], values["contact"], values["distance_pred"], values["distance_gt"],
        values["root_pred"], values["root_gt"],
    )
    metrics["loss"] = float(np.mean(losses)); metrics["num_windows"] = len(dataset)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
