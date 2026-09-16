"""Train/validate separately from explicit evaluation; resumable epoch boundaries."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import construct
from .components import ContactMetrics
from .data import ShardShuffleSampler


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def signature(cfg, manifest_sha):
    # Relocation/device and longer training are allowed; scientific changes are not resume.
    value = {k: cfg[k] for k in ("schema_version", "kind", "seed", "data", "model", "objective", "optimizer", "metrics")}
    value["batch_size"] = cfg["trainer"]["batch_size"]
    value["grad_clip"] = cfg["trainer"]["grad_clip"]
    value["manifest_sha256"] = manifest_sha
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment(cfg):
    project = Path(__file__).resolve().parents[3]
    def git(*args):
        p = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    source = hashlib.sha256()
    package = Path(__file__).resolve().parents[1]
    for p in sorted(package.rglob("*.py")):
        source.update(str(p.relative_to(package)).encode()); source.update(p.read_bytes())
    for name in ("data", "model", "objective"):
        module = importlib.import_module(cfg[name]["factory"].split(":")[0])
        source.update(cfg[name]["factory"].encode())
        source.update(Path(module.__file__).read_bytes())
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
            "training_source_sha256": source.hexdigest()}


def setup(cfg):
    torch.set_num_threads(cfg["trainer"]["torch_threads"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(cfg["trainer"]["deterministic"])
    device = torch.device(cfg["trainer"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set trainer.device=cpu for a CPU run")
    return device


def prepare(cfg, splits):
    spec = copy.deepcopy(cfg["data"])
    spec.setdefault("kwargs", {})["root"] = cfg["paths"]["data_root"]
    adapter = construct(spec)
    model = construct(cfg["model"])
    audit = adapter.audit(cfg["kind"], splits=splits, input_requirements=getattr(model, "required_inputs", ()))
    # Catch channel/target configuration errors in preflight, before creating a run.
    sample = adapter.dataset(splits[0])[0]
    model.eval()
    with torch.no_grad():
        outputs = model({k: v[None] for k, v in sample["inputs"].items()})
        loss = construct(cfg["objective"])(outputs, {k: v[None] for k, v in sample["targets"].items()})
        if not torch.isfinite(loss):
            raise ValueError("Preflight produced non-finite loss")
    model.train()
    return adapter, model, audit


def move(batch, device):
    return {group: {k: v.to(device) for k, v in values.items()} for group, values in batch.items()}


def evaluate(model, loader, device, metric_cfg):
    metrics = ContactMetrics(**metric_cfg)
    regions = {}
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            outputs = model(batch["inputs"])
            metrics.update(outputs, batch["targets"])
            for region in batch["metadata"]["region"].unique().tolist():
                select = batch["metadata"]["region"] == region
                meter = regions.setdefault(region, ContactMetrics(**metric_cfg))
                meter.update({k: v[select] for k, v in outputs.items()}, {k: v[select] for k, v in batch["targets"].items()})
    return {**metrics.compute(), "per_region": {str(k): meter.compute() for k, meter in regions.items() if meter.count}}


def train(cfg, resume=None):
    device = setup(cfg)
    adapter, model, audit = prepare(cfg, ("train", "val"))
    model = model.to(device)
    objective = construct(cfg["objective"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    options = {"batch_size": cfg["trainer"]["batch_size"], "num_workers": cfg["trainer"]["num_workers"]}
    train_data = adapter.dataset("train")
    train_loader = DataLoader(train_data, sampler=ShardShuffleSampler(train_data, generator), generator=generator, **options)
    val_loader = DataLoader(adapter.dataset("val"), shuffle=False, **options)
    identity = signature(cfg, adapter.sha256)
    metadata = environment(cfg)
    start, best, history = 0, -float("inf"), []
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("signature") != identity:
            raise ValueError("Resume config/data differ; use a new run for a new experiment")
        if checkpoint["environment"]["training_source_sha256"] != metadata["training_source_sha256"]:
            raise ValueError("Training code changed; resume requires the same source revision")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, history = checkpoint["epoch"], checkpoint["best_score"], checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if cfg["trainer"]["epochs"] <= start:
            raise ValueError("Requested epochs already completed")
    output = Path(cfg["paths"]["run_root"]) / cfg["name"]
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use this run's last.pt; relocate the complete run folder to continue elsewhere")
    output.mkdir(parents=True, exist_ok=bool(resume))
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume and (output / "last.pt").exists() and Path(resume).resolve() != (output / "last.pt").resolve():
            raise ValueError("Refusing to overwrite a different run; choose a new name")
        atomic_json(output / "config.resolved.json", cfg)
        atomic_json(output / "audit.json", audit)
        atomic_json(output / "environment.json", metadata)
        for epoch in range(start + 1, cfg["trainer"]["epochs"] + 1):
            model.train(); total_loss = samples = 0
            for raw in train_loader:
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(batch["inputs"]), batch["targets"])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"]["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                n = batch["targets"]["contact"].shape[0]
                total_loss += float(loss.detach()) * n; samples += n
            val = evaluate(model, val_loader, device, cfg["metrics"])
            record = {"epoch": epoch, "train_loss": total_loss / samples, "val": val}
            history.append(record)
            score = val["average_precision_histogram"]
            improved = score > best
            best = max(best, score)
            numpy_rng = np.random.get_state()
            checkpoint = {"format_version": 1, "epoch": epoch, "signature": identity, "config": cfg,
                          "manifest_sha256": adapter.sha256, "environment": metadata,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_score": best,
                          "history": history, "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                          "python_rng": random.getstate(), "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            if improved:
                atomic_checkpoint(output / "best.pt", checkpoint)
            atomic_checkpoint(output / "last.pt", checkpoint)
            atomic_json(output / "history.json", history)
            print(json.dumps(record), flush=True)
        return {"run": str(output), "epochs": cfg["trainer"]["epochs"], "best_val_average_precision_histogram": best, "kind": cfg["kind"]}


def evaluate_checkpoint(cfg, checkpoint_path, split):
    device = setup(cfg)
    adapter, model, audit = prepare(cfg, (split,))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != signature(cfg, adapter.sha256):
        raise ValueError("Evaluation config/data differ from the checkpoint")
    if checkpoint["environment"]["training_source_sha256"] != environment(cfg)["training_source_sha256"]:
        raise ValueError("Evaluation source differs from checkpoint; explicitly version the experiment")
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(adapter.dataset(split), batch_size=cfg["trainer"]["batch_size"], num_workers=cfg["trainer"]["num_workers"])
    return {"kind": cfg["kind"], "split": split, "epoch": checkpoint["epoch"], "audit": audit,
            "metrics": evaluate(model.to(device), loader, device, cfg["metrics"])}
