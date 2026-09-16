#!/usr/bin/env python3
"""Stage-A geometry-only baseline with valid PROX supervision.

The current PROX relation assets do not contain usable per-vertex binary
contact: the 2 cm teacher label is ROI-level and every vertex in one foot ROI
shares it.  This script therefore evaluates the two labels that PROX actually
supports: signed SDF proximity per vertex and one contact state per foot ROI.
It intentionally never reads RGB and is a local-geometry mechanism baseline,
not a result for a forward Gaussian/RGB front end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.stagea import LocalRelationEncoder


class Relations(Dataset):
    def __init__(self, manifests: list[str]):
        self.sequences: list[tuple[np.ndarray, ...]] = []
        self.index: list[tuple[int, int, int]] = []
        for manifest_path in manifests:
            manifest_file = Path(manifest_path)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            with np.load(manifest_file.parent / manifest["payload"], allow_pickle=False) as data:
                valid = data["scene_point_valid"] if "scene_point_valid" in data else np.ones(data["scene_point_local"].shape[:-1], np.float32)
                arrays = tuple(np.asarray(x, dtype=np.float32) for x in (
                    data["roi_vertex_local"], data["roi_vertex_normal_local"],
                    data["scene_point_local"], data["scene_normal_local"], valid,
                    data["vertex_proximity"], data["teacher_contact"],
                ))
            sequence_id = len(self.sequences)
            self.sequences.append(arrays)
            for frame in range(arrays[0].shape[0]):
                for foot in range(2):
                    self.index.append((sequence_id, frame, foot))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        sequence, frame, foot = self.index[index]
        a = self.sequences[sequence]
        return (
            torch.from_numpy(a[0][frame, foot]).float(),
            torch.from_numpy(a[1][frame, foot]).float(),
            torch.from_numpy(a[2][frame, foot]).float(),
            torch.from_numpy(a[3][frame, foot]).float(),
            torch.from_numpy(a[4][frame, foot]).float(),
            torch.from_numpy(a[5][frame, foot]).float(),
            torch.tensor(a[6][frame, foot], dtype=torch.float32),
        )


def evaluate(network: LocalRelationEncoder, loader: DataLoader, device: torch.device) -> dict[str, float]:
    network.eval()
    signed_error: list[torch.Tensor] = []
    absolute_error: list[torch.Tensor] = []
    predicted_contact: list[torch.Tensor] = []
    target_contact: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            b = [item.to(device) for item in batch]
            output = network(*b[:5])
            proximity = output["proximity"]
            # Logit pooling creates one semantic contact state for each foot
            # ROI; it deliberately does not pretend the labels are dense.
            roi_logits = output["contact_logits"].mean(dim=1)
            signed_error.append((proximity - b[5]).flatten().cpu())
            absolute_error.append((proximity.abs() - b[5].abs()).abs().flatten().cpu())
            predicted_contact.append((roi_logits.sigmoid() >= 0.5).cpu())
            target_contact.append((b[6] >= 0.5).cpu())
    signed = torch.cat(signed_error)
    absolute = torch.cat(absolute_error)
    pred = torch.cat(predicted_contact)
    target = torch.cat(target_contact)
    true_positive = (pred & target).sum().item()
    false_positive = (pred & ~target).sum().item()
    false_negative = (~pred & target).sum().item()
    return {
        "signed_proximity_mae_m": float(signed.abs().mean()),
        "absolute_proximity_mae_m": float(absolute.mean()),
        "proximity_rmse_m": float(torch.sqrt((signed.square()).mean())),
        "roi_contact_f1": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "roi_contact_precision": true_positive / max(true_positive + false_positive, 1),
        "roi_contact_recall": true_positive / max(true_positive + false_negative, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: Relations(paths) for name, paths in split["splits"].items()}
    loaders = {name: DataLoader(dataset, batch_size=args.batch, shuffle=name == "train", num_workers=0) for name, dataset in datasets.items()}
    network = LocalRelationEncoder().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-4)
    positives = sum(item[6].item() for item in datasets["train"])
    pos_weight = torch.tensor([(len(datasets["train"]) - positives) / max(positives, 1)], device=device)
    best: dict | None = None
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        network.train()
        losses = []
        for batch in loaders["train"]:
            b = [item.to(device) for item in batch]
            output = network(*b[:5])
            proximity_loss = torch.nn.functional.smooth_l1_loss(output["proximity"], b[5])
            roi_logits = output["contact_logits"].mean(dim=1)
            contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(roi_logits, b[6], pos_weight=pos_weight)
            loss = proximity_loss + 0.5 * contact_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(network, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}
        history.append(record)
        # Continuous geometry is the primary valid PROX target.
        if best is None or validation["signed_proximity_mae_m"] < best["signed_proximity_mae_m"]:
            best = {**record, "state": {key: value.detach().cpu() for key, value in network.state_dict().items()}}
    assert best is not None
    network.load_state_dict(best.pop("state"))
    result = {
        "kind": "stagea_geometry_only_proximity_roi_contact",
        "scope": "oracle PROX local-relation mechanism baseline; no RGB token, no forward Gaussian input, no controller",
        "split": str(args.split),
        "train_samples": len(datasets["train"]), "val_samples": len(datasets["val"]), "test_samples": len(datasets["test"]),
        "selection": "minimum validation signed proximity MAE",
        "best_validation": best,
        "test": evaluate(network, loaders["test"], device),
        "history": history,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(network.state_dict(), args.output / "best.pt")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation": best, "test": result["test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
