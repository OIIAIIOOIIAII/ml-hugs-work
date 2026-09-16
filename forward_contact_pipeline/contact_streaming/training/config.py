"""Small explicit configuration layer: inheritance, overrides and local roots."""
from __future__ import annotations

import copy
import importlib
import json
import os
import re
from pathlib import Path

import yaml


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def read_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    parent = value.pop("extends", None)
    return merge(read_config(path.parent / parent, (*stack, path)), value) if parent else value


def load_config(path, paths_path, overrides=()):
    cfg = read_config(path)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("Overrides must use dotted.key=YAML_VALUE")
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown override: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown override: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    paths_path = Path(paths_path).resolve()
    paths = read_config(paths_path)
    roots = {}
    for key in ("data_root", "run_root"):
        value = str(paths[key])
        for name in re.findall(r"\$\{([^}]+)\}", value):
            if name not in os.environ:
                raise ValueError(f"Set {name} or edit your local paths YAML")
            value = value.replace("${" + name + "}", os.environ[name])
        root = Path(value).expanduser()
        roots[key] = str((paths_path.parent / root).resolve())
    if cfg.get("schema_version") != 1 or cfg.get("kind") not in {"experiment", "smoke"}:
        raise ValueError("Expected schema_version=1 and kind=experiment or smoke")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(cfg["name"])):
        raise ValueError("Run name must be a single safe directory name")
    for key in ("epochs", "batch_size", "torch_threads"):
        if not isinstance(cfg["trainer"][key], int) or cfg["trainer"][key] < 1:
            raise ValueError(f"trainer.{key} must be positive")
    if cfg["trainer"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    cfg["paths"] = roots
    # Verify that checkpoints/config snapshots are JSON-compatible.
    json.dumps(cfg, allow_nan=False)
    return cfg


def construct(spec):
    """Factories receive keyword arguments; adding an experiment needs no runner edit."""
    module, separator, name = spec["factory"].partition(":")
    if not separator:
        raise ValueError("Component factory must be 'python.module:callable'")
    return getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))


def inside(root, relative):
    """Cache references are portable relative paths, never machine absolute paths."""
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Cache path must stay under its root: {relative}")
    return path
