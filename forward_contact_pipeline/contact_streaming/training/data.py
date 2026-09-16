"""Versioned numeric cache adapter; targets never enter a model's input mapping."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import inside

GEOMETRY = ("vertex_local", "vertex_normal", "point_local", "point_normal", "point_valid")
OPTIONAL_INPUTS = ("rgb_tokens", "gaussian_features")
TARGETS = ("contact", "contact_valid", "proximity", "proximity_valid")


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_arrays(inputs, targets):
    missing = set(GEOMETRY) - inputs.keys()
    if missing or not {"contact", "contact_valid"} <= targets.keys():
        raise ValueError(f"Missing geometry {sorted(missing)} or contact/valid targets")
    if set(inputs) - set(GEOMETRY + OPTIONAL_INPUTS) or set(targets) - set(TARGETS):
        raise ValueError("Unrecognized cache fields; version the adapter instead of silently consuming them")
    v = inputs["vertex_local"]
    p = inputs["point_local"]
    if v.ndim != 4 or v.shape[-1] != 3 or min(v.shape[:-1]) < 1:
        raise ValueError("vertex_local must be [frames, regions, vertices, 3]")
    shape = v.shape[:-1]
    if p.ndim != 5 or p.shape[:3] != shape or p.shape[-1] != 3 or p.shape[-2] < 1:
        raise ValueError("point_local must be [frames, regions, vertices, neighbors, 3]")
    expected = {"vertex_normal": v.shape, "point_normal": p.shape, "point_valid": p.shape[:-1]}
    for key, dimensions in expected.items():
        if inputs[key].shape != dimensions:
            raise ValueError(f"Invalid shape: {key}")
    for group in (inputs, targets):
        for key, array in group.items():
            if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
                raise ValueError(f"Non-numeric field: {key}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values: {key}; masked slots must also be finite")
    for key, array in targets.items():
        if array.shape != shape:
            raise ValueError(f"Invalid target shape: {key}")
    for key, array in {"point_valid": inputs["point_valid"], **{k: v for k, v in targets.items() if k.endswith("valid") or k == "contact"}}.items():
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{key} must be binary")
    if ("proximity" in targets) != ("proximity_valid" in targets):
        raise ValueError("proximity and proximity_valid must be supplied together")
    if not targets["contact_valid"].any():
        raise ValueError("Shard has no valid contact supervision")
    for key, mask in (("vertex_normal", np.ones(shape, bool)), ("point_normal", inputs["point_valid"].astype(bool))):
        norms = np.linalg.norm(inputs[key], axis=-1)[mask]
        if not np.allclose(norms, 1, atol=1e-3):
            raise ValueError(f"Valid {key} must be unit normals")
    if "rgb_tokens" in inputs:
        x = inputs["rgb_tokens"]
        if x.ndim != 4 or x.shape[:2] != shape[:2] or min(x.shape[-2:]) < 1:
            raise ValueError("rgb_tokens must be [frames, regions, tokens, channels]")
    if "gaussian_features" in inputs:
        x = inputs["gaussian_features"]
        if x.ndim != 5 or x.shape[:-1] != p.shape[:-1] or x.shape[-1] < 1:
            raise ValueError("gaussian_features must match point neighbors")
    return shape


class NumericCache:
    """One index can describe RICH, PROX or future datasets using the same schema.

    Shard NPZs to a bounded number of frames. Each loader worker keeps at most
    cache_shards decoded input/target pairs rather than loading a dataset in RAM.
    """
    def __init__(self, root, index, cache_shards=1, max_frames_per_shard=32):
        self.root = Path(root).resolve()
        self.path = inside(self.root, index)
        self.index = json.loads(self.path.read_text())
        self.sha256 = digest(self.path)
        self.cache_shards = int(cache_shards)
        self.max_frames_per_shard = int(max_frames_per_shard)
        if self.cache_shards < 1 or self.index.get("schema_version") != 1:
            raise ValueError("Invalid cache size or index schema")
        self.records = self.index["records"]
        if not self.records:
            raise ValueError("Empty index")
        seen, ownership = set(), {k: {} for k in ("scene_id", "sequence_id", "subject_id", "input", "target")}
        self.shapes = {}
        for row in self.records:
            if row["id"] in seen or row["split"] not in {"train", "val", "test"}:
                raise ValueError("Duplicate shard id or invalid split")
            seen.add(row["id"])
            for key, mapping in ownership.items():
                value = row[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Missing split identity: {key}")
                if key in ("input", "target"):
                    value = str(inside(self.root, value))
                if value in mapping and mapping[value] != row["split"]:
                    raise ValueError(f"Split leakage in {key}: {value}")
                mapping[value] = row["split"]

    def audit(self, kind, splits=("train", "val"), input_requirements=()):
        provenance = self.index.get("provenance", {})
        if kind == "experiment":
            required = {"kind": "real", "input_source": "frozen_backbone", "units": "m", "coordinate_frame": "contact_local", "temporal_context": "current_and_past", "topology": "smplx"}
            if any(provenance.get(k) != v for k, v in required.items()):
                raise ValueError("Real experiments require a frozen, causal, contact-local SMPL-X cache")
            for key in ("backbone_id", "backbone_revision", "feature_version", "label_source"):
                if not provenance.get(key):
                    raise ValueError(f"Missing provenance: {key}")
            report = json.loads(inside(self.root, self.index["audit_report"]).read_text())
            if report.get("manifest_sha256") != self.sha256:
                raise ValueError("Audit report is stale or belongs to another manifest")
            checks = report.get("checks", {})
            if not all(checks.get(k) is True for k in ("coordinates", "topology", "frame_alignment", "label_distribution", "frontend_geometry")):
                raise ValueError("Real-data entry checks have not passed")
        elif provenance.get("kind") != "synthetic":
            raise ValueError("Smoke mode accepts explicitly synthetic caches only")
        counts = {}
        compatible = None
        for split in splits:
            rows = [r for r in self.records if r["split"] == split]
            if not rows:
                raise ValueError(f"Missing split: {split}")
            positive = total = samples = 0
            for row in rows:
                inputs, targets = self.load(row)
                shape = validate_arrays(inputs, targets)
                if shape[0] > self.max_frames_per_shard:
                    raise ValueError("Shard exceeds frame budget; split caches before training")
                for key in input_requirements:
                    if key not in inputs:
                        raise ValueError(f"Model requires cached input {key}")
                feature_shape = {k: tuple(x.shape[2:]) for k, x in inputs.items()}
                feature_shape["target_fields"] = tuple(sorted(targets))
                if compatible is not None and feature_shape != compatible:
                    raise ValueError("Shards must share region/vertex feature shapes for batching")
                compatible = feature_shape
                if kind == "experiment":
                    for key in ("input", "target"):
                        if digest(inside(self.root, row[key])) != row.get(key + "_sha256"):
                            raise ValueError(f"Cache content changed since audit: {row['id']} {key}")
                self.shapes[row["id"]] = shape
                valid = targets["contact_valid"].astype(bool)
                positive += int(targets["contact"][valid].sum())
                total += int(valid.sum())
                samples += shape[0] * shape[1]
            if positive == 0 or positive == total:
                raise ValueError(f"{split} has only one contact class; cannot train/evaluate discrimination")
            counts[split] = {"samples": samples, "positive": positive, "valid_vertices": total, "positive_rate": positive / total}
        return {"manifest_sha256": self.sha256, "kind": kind, "splits": counts}

    def load(self, row):
        return read_npz(inside(self.root, row["input"])), read_npz(inside(self.root, row["target"]))

    def dataset(self, split):
        return ShardDataset(self, split)


class ShardDataset(Dataset):
    def __init__(self, adapter, split):
        self.adapter = adapter
        self.rows = [r for r in adapter.records if r["split"] == split]
        self.items = [(i, t, roi) for i, r in enumerate(self.rows) for t in range(adapter.shapes[r["id"]][0]) for roi in range(adapter.shapes[r["id"]][1])]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, frame, region = self.items[index]
        if i not in self.cache:
            self.cache[i] = self.adapter.load(self.rows[i])
            while len(self.cache) > self.adapter.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        inputs, targets = self.cache[i]
        convert = lambda group: {k: torch.from_numpy(np.asarray(v[frame, region], dtype=np.float32).copy()) for k, v in group.items()}
        return {"inputs": convert(inputs), "targets": convert(targets), "metadata": {"region": torch.tensor(region)}}


class ShardShuffleSampler(Sampler):
    """Shuffle shard order and samples within each shard, avoiding NPZ reload per sample."""
    def __init__(self, dataset, generator):
        self.dataset, self.generator = dataset, generator
        self.groups = {}
        for index, (shard, _, _) in enumerate(dataset.items):
            self.groups.setdefault(shard, []).append(index)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for shard in torch.randperm(len(self.groups), generator=self.generator).tolist():
            indices = self.groups[shard]
            for position in torch.randperm(len(indices), generator=self.generator).tolist():
                yield indices[position]
