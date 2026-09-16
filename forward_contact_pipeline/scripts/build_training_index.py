#!/usr/bin/env python3
"""Index explicitly joined input/label shards; never certify geometry automatically.

The catalog supplies provenance and scene/sequence/subject/split identities. RICH
raw-to-cache export must align frames and SMPL-X vertex IDs before this step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_streaming.training.config import inside
from contact_streaming.training.data import NumericCache, digest, read_npz, validate_arrays


def build(root, catalog_path, output):
    root = Path(root).resolve()
    output = inside(root, output)
    if output.exists():
        raise ValueError("Index already exists; version the cache instead of overwriting it")
    catalog = json.loads(Path(catalog_path).read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Catalog schema_version must be 1")
    for record in catalog["records"]:
        inputs = inside(root, record["input"])
        targets = inside(root, record["target"])
        validate_arrays(read_npz(inputs), read_npz(targets))
        record["input_sha256"], record["target_sha256"] = digest(inputs), digest(targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate identities without leaving a usable-looking index on failure.
    temporary = output.with_suffix(".pending.json")
    if temporary.exists():
        raise ValueError("Pending index already exists")
    try:
        temporary.write_text(json.dumps(catalog, indent=2) + "\n")
        NumericCache(root, str(temporary.relative_to(root)))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"index": str(output), "manifest_sha256": digest(output), "shards": len(catalog["records"]),
            "geometry_audit": "not performed; real training requires an independent audit report"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", required=True, help="New index path relative to data-root")
    args = parser.parse_args()
    print(json.dumps(build(args.data_root, args.catalog, args.output), indent=2))
