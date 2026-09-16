#!/usr/bin/env python3
"""Prepare a reviewable source-only Git path list; never stage, commit or push.

This is a pre-publication guard, not a claim that all historical Git objects have
been scanned. Data/weights/cache transfer belongs to the asset migration plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("hugs/", "scripts/", "tools/", "cfg_files/", "migration/", "transmission_system/", "idea/", "ref/", "test md/",
                "forward_contact_pipeline/contact_streaming/", "forward_contact_pipeline/scripts/",
                "forward_contact_pipeline/reports/",
                "forward_contact_pipeline/configs/", "forward_contact_pipeline/tests/", "forward_contact_pipeline/baseline/")
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".json", ".patch", ".tsv", ".txt"}
GENERATED_PARTS = {"runs", "outputs", "results", "checkpoints", "cache", "__pycache__", ".ipynb_checkpoints", "datasets", "data"}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{25,}\b"),
    "github_token": re.compile(rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def audit(project):
    project = Path(project)
    def names(*args):
        result = subprocess.run(["git", "-C", str(project), "ls-files", "-z", *args], capture_output=True, check=True)
        return set(result.stdout.decode().split("\0")) - {""}
    tracked = names("--cached")
    untracked = names("--others", "--exclude-standard")
    candidates, blockers, absolute_paths, existing_assets = [], [], [], []
    for name in sorted(tracked | untracked):
        p = project / name
        if p.is_dir():  # Gitlinks are represented by .gitmodules/dependency pins.
            continue
        if not p.exists():
            if name in tracked:
                blockers.append({"path": name, "reason": "tracked deletion requires review"})
            continue
        if name in tracked and p.suffix not in SOURCE_SUFFIXES and p.name not in {".gitignore", ".gitmodules", "requirements.txt", "LICENSE", "ACKNOWLEDGEMENTS", "CODE_OF_CONDUCT", "CONTRIBUTING"}:
            existing_assets.append({"path": name, "bytes": p.stat().st_size})
            continue  # Existing assets already travel with Git; do not inspect image bytes.
        if name not in tracked:
            root_doc = len(p.relative_to(project).parts) == 1 and p.suffix in {".md", ".py", ".sh"}
            pipeline_doc = p.parent == project / "forward_contact_pipeline" and (
                p.suffix in {".md", ".toml", ".py"} or (p.name.startswith("requirements") and p.suffix == ".txt"))
            source = name.startswith(SOURCE_ROOTS) and p.suffix in SOURCE_SUFFIXES
            generated = GENERATED_PARTS.intersection(p.relative_to(project).parts) and not name.startswith(("cfg_files/", "forward_contact_pipeline/configs/"))
            if not (root_doc or pipeline_doc or source) or generated:
                continue
        if p.is_symlink() or "\n" in name:
            blockers.append({"path": name, "reason": "symlink or unusual filename needs explicit review"}); continue
        if p.stat().st_size > 5 * 1024 * 1024:
            blockers.append({"path": name, "reason": "file over 5 MiB is outside the code migration budget"}); continue
        blob = p.read_bytes()
        findings = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(blob)]
        if findings:
            blockers.append({"path": name, "reason": "possible secret", "patterns": findings}); continue
        # Do not echo source lines: a path finding is enough for portability review.
        if p.suffix in {".py", ".sh", ".yaml", ".yml"}:
            lines = [i for i, line in enumerate(blob.splitlines(), 1) if re.search(rb"/(?:workspace/|home/admin/|mnt/)", line)]
            if lines:
                absolute_paths.append({"path": name, "lines": lines})
        candidates.append(name)
    return {"schema_version": 1, "source_files": candidates, "source_file_count": len(candidates),
            "untracked_source_count": sum(n in untracked for n in candidates),
            "existing_tracked_assets": existing_assets,
            "blockers": blockers, "legacy_absolute_paths": absolute_paths,
            "note": "No staging/push performed. Existing Git history, remote visibility and licenses still require review before publishing."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(PROJECT)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output / "source.paths").write_text("\n".join(result["source_files"]) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"source_files", "legacy_absolute_paths"}}, ensure_ascii=False))
    print(f"Historical files with machine paths: {len(result['legacy_absolute_paths'])}")
    raise SystemExit(bool(result["blockers"]))
