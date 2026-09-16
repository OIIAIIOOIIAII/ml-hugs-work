"""Join completed RICH assets without decoding pixels or inventing frontend inputs.

All paths in published artifacts are relative to the RICH root. Samples index
GT supervision and raw images; they are deliberately not NumericCache records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import stat
import time

import numpy as np

from .rich_assets import member_path, save_json, sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def inside(root, relative):
    path = root / member_path(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path


def write_line(stream, record):
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def validate_image_identities(rows):
    pattern = re.compile(r"([^/]+)/cam_(\d+)/(\d+)_(\d+)\.(?:jpg|jpeg)")
    for row in rows:
        match = pattern.fullmatch(row["path"])
        if match is None:
            raise ValueError(f"Unrecognized image identity: {row['path']}")
        sequence, camera, frame, file_camera = match.groups()
        actual = sequence, int(camera), int(frame), int(file_camera)
        expected = row["sequence"], row["camera_id"], row["frame_id"], row["camera_id"]
        if actual != expected:
            raise ValueError(f"Filename/frame/camera identity mismatch: {row['path']}")


def freeze_splits(tracks, config):
    mapping = config["capture_splits"]
    if set(mapping) != {t["capture"] for t in tracks} or set(mapping.values()) != {"train", "val"}:
        raise ValueError("Assign every capture exactly once to train or val")
    seen = {key: {} for key in ("capture", "scene", "sequence", "subject")}
    records = []
    track_keys = set()
    for track in sorted(tracks, key=lambda t: (t["sequence"], t["subject"])):
        key = track["sequence"], track["subject"]
        if key in track_keys:
            raise ValueError("Duplicate subject track")
        track_keys.add(key)
        split = mapping[track["capture"]]
        record = {k: track[k] for k in ("capture", "scan", "sequence", "subject", "gender")}
        record.update(scene=f"{track['capture']}/{track['scan']}", split=split)
        for field, assigned in seen.items():
            value = record[field]
            if value in assigned and assigned[value] != split:
                raise ValueError(f"Split leakage: {field}={value}")
            assigned[value] = split
        records.append(record)
    return {"schema_version": 1, "name": config["name"], "source_split": "official_train",
            "purpose": "internal development; not official validation/test",
            "isolation": {key: "verified_disjoint" for key in seen}, "tracks": records}


@lru_cache(maxsize=4096)
def image_parent(root, relative):
    # Resolve each camera directory once instead of walking the NAS mount's
    # ancestors for every one of its images. Final files must be regular files.
    return inside(root, relative).resolve()


def image_header(root, row, cached=None):
    # Pillow reads JPEG markers and EXIF lazily. Never call load/convert/transpose.
    from PIL import Image
    relative = member_path("extracted/train/" + row["path"])
    path = image_parent(root, str(relative.parent)) / relative.name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Image must be a regular file: {row['path']}")
    if info.st_size != row["bytes"]:
        raise ValueError(f"Image size changed: {row['path']}")
    if (cached is not None and cached.get("source_sha256") == row["sha256"]
            and cached.get("width") == row["width"] and cached.get("height") == row["height"]
            and cached.get("mtime_ns") == info.st_mtime_ns and cached.get("orientation") in range(1, 9)):
        return cached
    with Image.open(path) as image:
        orientation = int(image.getexif().get(274, 1))
        if image.format != "JPEG" or image.size != (row["width"], row["height"]):
            raise ValueError(f"Image header disagrees with index: {row['path']}")
    if orientation not in range(1, 9):
        raise ValueError(f"Invalid EXIF orientation: {row['path']}")
    return {"path": row["path"], "source_sha256": row["sha256"],
            "width": row["width"], "height": row["height"],
            "orientation": orientation, "mtime_ns": info.st_mtime_ns}


def audit_cameras(rows, headers, tracks, calibration):
    by_sequence = {t["sequence"]: t for t in tracks}
    groups = defaultdict(list)
    for row, header in zip(rows, headers):
        track = by_sequence[row["sequence"]]
        groups[(track["capture"], track["scan"], row["camera_id"])].append((row, header))
    result = {}
    for (capture, scan, camera), values in sorted(groups.items()):
        dimensions = {(r["width"], r["height"]) for r, _ in values}
        if len(dimensions) != 1:
            raise ValueError(f"Camera dimensions vary: {capture}/{camera}")
        width, height = next(iter(dimensions))
        orientations = Counter(str(h["orientation"]) for _, h in values)
        entry = {"capture": capture, "scan": scan, "camera_id": camera,
                 "image_count": len(values), "raw_width": width, "raw_height": height,
                 "exif_orientations": dict(orientations)}
        raw = calibration["captures"][f"{capture}/{scan}"]["cameras"].get(str(camera))
        if raw is None:
            entry["status"] = "excluded_missing_calibration"
        else:
            k = np.asarray(raw["Intrinsics"], dtype=np.float64)
            e = np.asarray(raw["CameraMatrix"], dtype=np.float64)
            distortion = np.asarray(raw["Distortion"], dtype=np.float64)
            if (k.shape != (3, 3) or e.shape != (3, 4) or not np.isfinite(k).all()
                    or not np.isfinite(e).all() or not np.isfinite(distortion).all()
                    or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
                raise ValueError(f"Invalid camera matrices: {capture}/{camera}")
            error = float(np.abs(e[:, :3] @ e[:, :3].T - np.eye(3)).max())
            if error > 1e-5 or abs(np.linalg.det(e[:, :3]) - 1) > 1e-5:
                raise ValueError(f"Invalid camera rotation: {capture}/{camera}")
            if not (0 <= k[0, 2] < width and 0 <= k[1, 2] < height):
                raise ValueError(f"Principal point outside raw image: {capture}/{camera}")
            if np.any(distortion != 0):
                raise ValueError("Nonzero distortion requires an explicit image transform adapter")
            entry.update(status="structurally_consistent", **raw,
                         rotation_orthogonality_max_error=error,
                         raw_pixel_transform=np.eye(3).tolist(),
                         intrinsics_scope="raw JPEG raster; frontend resize/crop requires an explicit transform")
        result[f"{capture}/{scan}/cam_{camera:02d}"] = entry
    return {"schema_version": 1, "cameras": result,
            "pixel_alignment_verified": False, "frontend_geometry_verified": False,
            "scope": "All image headers, dimensions, EXIF and camera matrix structure; no pixel decoding"}


def index_shards(root, state, tracks, workers):
    expected = {(t["sequence"], t["subject"], f) for t in tracks for f in t["frames"]}
    records = {}

    def inspect(shard):
        path = inside(root, "processed/annotations_v1/" + shard["path"])
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"Annotation hash changed: {shard['path']}")
        with np.load(path, allow_pickle=False) as arrays:
            frames = arrays["frame_ids"]
            labels, valid = arrays["contact_smplx"], arrays["contact_smplx_valid"]
            smpl = arrays["contact_smpl_valid"]
            if (frames.tolist() != shard["frames"] or labels.shape != (len(frames), 10475)
                    or valid.shape != labels.shape or valid.dtype != bool or labels.dtype != bool
                    or smpl.shape != (len(frames), 6890)):
                raise ValueError(f"Invalid annotation schema: {shard['path']}")
            counts = valid.sum(axis=1)
            positives = (labels & valid).sum(axis=1)
            missing = [int(f) for f, count in zip(frames, counts) if count == 0]
            if missing != shard["missing_smplx_color_frames"]:
                raise ValueError("SMPL-X validity differs from annotation state")
            return [(int(f), int(count), int(positive), int(smpl[i].sum()))
                    for i, (f, count, positive) in enumerate(zip(frames, counts, positives))]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (shard, values) in enumerate(zip(state["shards"], pool.map(inspect, state["shards"]))):
            for offset, (frame, valid, positive, smpl_valid) in enumerate(values):
                key = shard["sequence"], shard["subject"], frame
                if key in records:
                    raise ValueError(f"Duplicate annotation frame: {key}")
                records[key] = {"path": "processed/annotations_v1/" + shard["path"],
                                "sha256": shard["sha256"], "row": offset, "frame_id": frame,
                                "smplx_valid_vertices": valid, "smplx_positive_vertices": positive,
                                "smpl_valid_vertices": smpl_valid}
            if (index + 1) % 100 == 0:
                print(f"Annotation shards verified: {index + 1}/{len(state['shards'])}", flush=True)
    if set(records) != expected or len(records) != state["completed_frames"]:
        raise ValueError("Annotation shard coverage differs from catalog")
    return records


def projection_summary(vertices, camera):
    e, k = np.asarray(camera["CameraMatrix"]), np.asarray(camera["Intrinsics"])
    points = vertices @ e[:, :3].T + e[:, 3]
    front = points[:, 2] > 1e-6
    pixels = points[front] @ k.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    inside_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < camera["raw_width"])
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera["raw_height"]))
    return {"front_fraction": float(front.mean()),
            "in_image_fraction": float(inside_image.sum() / len(vertices)),
            "depth_range_m": [float(points[:, 2].min()), float(points[:, 2].max())]}


def choose_clip(rows, spec):
    selected = sorted((r for r in rows if all(r[k] == spec[k] for k in ("split", "sequence", "subject", "camera_id"))),
                      key=lambda r: r["frame_id"])
    run = []
    for row in selected:
        if run and row["frame_id"] != run[-1]["frame_id"] + 1:
            run = []
        run.append(row)
        if len(run) == spec["length"]:
            return {**spec, "frames": run, "frame_selection": "first consecutive eligible run; no label-value selection"}
    raise ValueError(f"No consecutive pilot clip: {spec}")


def prepare_dataset(root, config_path, output, workers=4):
    root, config_path, output = Path(root).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "processed") or workers < 1:
        raise ValueError("Output must be inside ROOT/processed; workers must be positive")
    config = read_json(config_path)
    if (config.get("schema_version") != 1 or config.get("source_split") != "official_train"
            or config.get("label_space") != "smplx_10475" or not config.get("require_official_camera")
            or not config.get("require_identity_exif")):
        raise ValueError("Unsupported dataset configuration")
    inputs = ["processed/images_v1/images.jsonl", "processed/annotations_v1/catalog.json",
              "processed/annotations_v1/state.json", "processed/annotations_v1/calibration.json",
              "processed/annotations_v1/smplx_topology.npz", "processing_logs/train_jpg.json"]
    sources = {name: sha256(root / name) for name in inputs}
    signature = hashlib.sha256(json.dumps({"sources": sources, "config": config}, sort_keys=True).encode()).hexdigest()
    if output.exists():
        previous = read_json(output / "report.json")
        if previous["source_signature"] != signature or previous["status"] != "complete":
            raise ValueError("Output source/config differs; use a new version")
        for name, digest in previous["artifacts_sha256"].items():
            if sha256(inside(output, name)) != digest:
                raise ValueError(f"Published artifact changed: {name}")
        return previous
    pending = output.with_name(output.name + ".pending")
    pending.mkdir(parents=True, exist_ok=True)
    state_path = pending / "progress.json"
    if state_path.exists() and read_json(state_path)["source_signature"] != signature:
        raise ValueError("Pending output source/config differs; use a new version")
    progress = {"status": "running", "source_signature": signature, "training_ready": False}
    save_json(state_path, progress)
    started = time.time()
    try:
        catalog = read_json(root / inputs[1]); state = read_json(root / inputs[2])
        calibration = read_json(root / inputs[3]); extraction = read_json(root / inputs[5])
        catalog_signature = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        if (state["status"] != "complete" or state["mode"] != "full"
                or state["catalog_signature"] != catalog_signature
                or sources[inputs[4]] != state["topology_sha256"]
                or extraction["status"] != "complete" or extraction["archive_integrity"] != "verified"):
            raise ValueError("Completed and verified source assets required")
        with (root / inputs[0]).open() as stream:
            rows = [json.loads(line) for line in stream]
        validate_image_identities(rows)
        rows.sort(key=lambda r: (r["sequence"], r["camera_id"], r["frame_id"], r["path"]))
        if (len(rows) != extraction["image_files"] or len({r["path"] for r in rows}) != len(rows)
                or len({(r['sequence'], r['camera_id'], r['frame_id']) for r in rows}) != len(rows)
                or sum(r["bytes"] for r in rows) != extraction["uncompressed_bytes"]):
            raise ValueError("Image index count/bytes/uniqueness check failed")
        tracks = catalog["tracks"]
        split = freeze_splits(tracks, config)
        save_json(pending / "splits.json", split)
        save_json(pending / "config.json", config)
        annotations = index_shards(root, state, tracks, min(workers, 4))
        header_cache = {}
        header_path = pending / "image_headers.jsonl"
        if header_path.exists():
            with header_path.open() as previous:
                for line in previous:
                    try:
                        cached = json.loads(line)
                    except json.JSONDecodeError:
                        if line.endswith("\n") or previous.read():
                            raise ValueError("Malformed pending header cache")
                        break  # A terminated writer may leave one partial final line.
                    if cached["path"] in header_cache:
                        raise ValueError("Duplicate path in pending header cache")
                    header_cache[cached["path"]] = cached
            print(f"Rechecking {len(header_cache)} cached headers against source hash, size and mtime", flush=True)
        headers = []
        with header_path.open("w") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            # Submit bounded chunks: avoid 261k simultaneous Futures on a shared host.
            for start in range(0, len(rows), 1024):
                for header in pool.map(lambda r: image_header(root, r, header_cache.get(r["path"])), rows[start:start + 1024]):
                    headers.append(header); write_line(stream, header)
                if start % 10240 == 0:
                    print(f"Image headers verified: {len(headers)}/{len(rows)} (no pixels decoded)", flush=True)
                    save_json(state_path, {**progress, "image_headers": len(headers), "updated": time.time()})
        cameras = audit_cameras(rows, headers, tracks, calibration)
        save_json(pending / "cameras.json", cameras)
        by_sequence = defaultdict(list)
        for track in tracks:
            by_sequence[track["sequence"]].append(track)
        counts = Counter(); split_counts = defaultdict(Counter); covered = set(); supervised = set()
        pilot_rows = []; projection_rows = []
        probe_keys = {(t['sequence'], t['subject'], f) for t in tracks
                      for f in {t['frames'][0], t['frames'][len(t['frames']) // 2], t['frames'][-1]}}
        with ExitStack() as stack:
            streams = {s: stack.enter_context((pending / f"samples_{s}.jsonl").open("w")) for s in ("train", "val")}
            excluded = stack.enter_context((pending / "excluded.jsonl").open("w"))
            for row, header in zip(rows, headers):
                matched = False
                for track in by_sequence[row["sequence"]]:
                    key = row["sequence"], track["subject"], row["frame_id"]
                    annotation = annotations.get(key)
                    if annotation is None:
                        continue
                    matched = True
                    split_name = config["capture_splits"][track["capture"]]
                    camera_key = f"{track['capture']}/{track['scan']}/cam_{row['camera_id']:02d}"
                    camera = cameras["cameras"][camera_key]
                    reasons = []
                    if row["camera_id"] not in track["camera_ids"]:
                        reasons.append("camera_outside_official_metadata")
                    if camera["status"] != "structurally_consistent":
                        reasons.append("missing_calibration")
                    if header["orientation"] != 1:
                        reasons.append("nonidentity_exif_requires_adapter")
                    if not reasons:
                        covered.add(key)
                    if annotation["smplx_valid_vertices"] == 0:
                        reasons.append("missing_smplx_contact_colors")
                    sample = {"sample_id": f"{key[0]}/{key[1]}/cam_{row['camera_id']:02d}/{key[2]:05d}",
                              "sequence": key[0], "subject": key[1], "frame_id": key[2], "split": split_name,
                              "scene": f"{track['capture']}/{track['scan']}", "capture": track["capture"],
                              "gender": track["gender"], "camera_id": row["camera_id"], "camera_key": camera_key,
                              "image": "extracted/train/" + row["path"], "image_sha256": row["sha256"],
                              "image_bytes": row["bytes"], "raw_size_wh": [row["width"], row["height"]],
                              "exif_orientation": header["orientation"], "annotation": annotation,
                              "label_space": "smplx_10475", "frontend_status": "not_exported"}
                    if reasons:
                        write_line(excluded, {**sample, "reasons": reasons})
                        counts["excluded_image_subject_pairs"] += 1
                        for reason in reasons:
                            counts["excluded_reason_" + reason] += 1
                    else:
                        write_line(streams[split_name], sample)
                        split_counts[split_name]["image_subject_samples"] += 1
                        supervised.add(key)
                        if any(row["sequence"] == p["sequence"] for p in config["pilot_clips"]):
                            pilot_rows.append(sample)
                    if key in probe_keys and not any(r != "missing_smplx_contact_colors" for r in reasons):
                        projection_rows.append(sample)
                if not matched:
                    counts["images_without_contact_annotation"] += 1
                    write_line(excluded, {"image": "extracted/train/" + row["path"],
                                          "reasons": ["no_contact_annotation_at_frame"]})
        if covered != set(annotations):
            raise ValueError(f"{len(set(annotations) - covered)} contact frames lack a calibrated identity-EXIF image")
        valid_keys = {k for k, v in annotations.items() if v["smplx_valid_vertices"] > 0}
        if supervised != valid_keys:
            raise ValueError("Valid SMPL-X frame coverage is incomplete")
        for t in tracks:
            name = config["capture_splits"][t["capture"]]
            split_counts[name]["tracks"] += 1
            split_counts[name]["contact_frames"] += len(t["frames"])
            split_counts[name]["valid_smplx_frames"] += sum((t['sequence'], t['subject'], f) in valid_keys for f in t['frames'])
        for name in split_counts:
            split_counts[name]["sequences"] = len({t['sequence'] for t in split['tracks'] if t['split'] == name})
        projections = []
        grouped = defaultdict(list)
        for row in projection_rows:
            grouped[row["annotation"]["path"]].append(row)
        for path, samples in grouped.items():
            with np.load(inside(root, path), allow_pickle=False) as arrays:
                vertices = arrays["vertices_multicam"]
                for sample in samples:
                    projections.append({"sample_id": sample["sample_id"], **projection_summary(
                        vertices[sample["annotation"]["row"]], cameras["cameras"][sample["camera_key"]])})
        save_json(pending / "projection_audit.json", {
            "scope": "first/middle/last GT frame per track projected numerically; no pixel or frontend accuracy validation",
            "pixel_alignment_verified": False, "samples": projections})
        save_json(pending / "pilot_clips.json", {"schema_version": 1, "kind": "raw_frontend_input_plan",
            "clips": [choose_clip(pilot_rows, spec) for spec in config["pilot_clips"]],
            "status": "not_run", "requirement": "causal frozen frontend; preserve real frame IDs and image transforms"})
        save_json(pending / "missing_contact_frames.json", catalog["excluded_missing_contact"])
        save_json(pending / "provenance.json", {"source_signature": signature, "sources_sha256": sources,
            "metadata": catalog["metadata"], "image_archive_sha256": extraction["archive_sha256"],
            "image_hash_scope": "Computed during verified extraction; this pass checks all file sizes and JPEG/EXIF headers",
            "annotation_hash_scope": "All annotation shards SHA256 re-read in this run",
            "coordinate_reference": "processed/annotations_v1/calibration.json",
            "path_base": "RICH root; relocatable"})
        # Detect changed source manifests during a long metadata scan.
        if any(sha256(root / path) != digest for path, digest in sources.items()):
            raise ValueError("Source manifests changed during preparation")
        report = {"schema_version": 1, "status": "complete", "source_signature": signature,
            "training_ready": False, "supervision_index_ready": True, "image_count": len(rows),
            "header_orientation_counts": dict(Counter(str(h["orientation"]) for h in headers)),
            "annotation_frames": len(annotations), "valid_smplx_frames": len(valid_keys),
            "missing_smplx_color_frames": len(annotations) - len(valid_keys),
            "body_only_frames": len(catalog["excluded_missing_contact"]),
            "calibrated_image_frame_coverage": len(covered), "splits": dict(split_counts),
            "counts": dict(counts), "elapsed_seconds": time.time() - started,
            "limitations": ["Official validation/test not downloaded", "Frontend features not exported",
                            "Frontend resize/crop camera adapter and independent geometry audit pending",
                            "Numerical projection sanity is not pixel correspondence or visibility validation",
                            "Raw official world coefficients retained, including ParkingLot1 rounding"],
            "artifacts_sha256": {p.name: sha256(p) for p in sorted(pending.iterdir())
                                 if p.is_file() and p.name not in {"progress.json", "report.json"}}}
        save_json(pending / "report.json", report)
        save_json(state_path, {**progress, "status": "complete", "updated": time.time()})
        pending.rename(output)
        return report
    except Exception as error:
        save_json(state_path, {**progress, "status": "failed", "error": str(error), "updated": time.time()})
        raise
