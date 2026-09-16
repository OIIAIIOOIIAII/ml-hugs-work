"""RICH GT annotation preparation. Does not produce frozen frontend inputs.

Only numeric mesh data is read; no image is opened or rendered. Raw archives
remain authoritative. Official SMPL (6890) and SMPL-X (10475) labels are distinct.
"""
from __future__ import annotations

import _codecs
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .rich_assets import member_path, save_json, sha256

VERSION = 1
BODY_SHAPES = {"betas": 10, "global_orient": 3, "transl": 3,
               "left_hand_pose": 12, "right_hand_pose": 12, "jaw_pose": 3,
               "leye_pose": 3, "reye_pose": 3, "expression": 10,
               "pose_embedding": 32, "body_pose": 63}


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
                   ("_codecs", "encode"): _codecs.encode}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = np.core.multiarray._reconstruct
            allowed[prefix, "scalar"] = np.core.multiarray.scalar
        if (module, name) not in allowed:
            raise ValueError(f"Non-numeric pickle global: {module}.{name}")
        return allowed[module, name]


def numeric_pickle(data):
    value = NumericUnpickler(io.BytesIO(data), encoding="latin1").load()
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected a numeric dictionary")
    for key, array in value.items():
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"Invalid numeric array: {key}")
    return value


def obj_mesh(data, expected_faces_hash=None):
    lines = data.splitlines()
    vertices = [line[2:] for line in lines if line.startswith(b"v ")]
    widths = {len(line.split()) for line in vertices}
    if len(vertices) != 10475 or widths not in ({3}, {6}):
        raise ValueError("Expected 10475 SMPL-X vertices with uniform XYZ or XYZRGB layout")
    width = next(iter(widths))
    values = np.fromstring(b" ".join(vertices).decode("ascii"), sep=" ").reshape(10475, width)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite mesh")
    valid = np.full(10475, width == 6, dtype=bool)
    green = np.zeros(10475, dtype=bool)
    if width == 6:
        green = np.all(np.isclose(values[:, 3:], [0, 1, 0], atol=1e-7, rtol=0), axis=1)
        gray = np.all(np.isclose(values[:, 3:], [.4, .4, .4], atol=1e-7, rtol=0), axis=1)
        if not np.all(green | gray):
            raise ValueError("Unknown official contact color; do not infer a label")
    face_lines = [line[2:] for line in lines if line.startswith(b"f ")]
    digest = hashlib.sha256(b"\n".join(face_lines)).hexdigest()
    faces = None
    if expected_faces_hash is not None:
        if digest != expected_faces_hash:
            raise ValueError("Contact face topology/order changed")
    else:
        faces = np.asarray([[int(x.split(b"/")[0]) - 1 for x in line.split()] for line in face_lines], dtype=np.int32)
        if faces.shape != (20908, 3) or faces.min() < 0 or faces.max() >= 10475:
            raise ValueError("Invalid SMPL-X triangular faces")
    return values[:, :3], green, valid, digest, faces


def ply_mesh(data):
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    expected = ["ply", "format binary_little_endian 1.0", "element vertex 10475",
                "property float x", "property float y", "property float z",
                "element face 20908", "property list uchar int vertex_indices", "end_header"]
    if [line for line in header if not line.startswith("comment ")] != expected:
        raise ValueError("Unsupported RICH body PLY layout")
    vertices = np.frombuffer(data, dtype="<f4", count=10475 * 3, offset=end).reshape(10475, 3)
    offset = end + vertices.nbytes
    faces = np.frombuffer(data, dtype=[("count", "u1"), ("v", "<i4", (3,))], count=20908, offset=offset)
    if len(data) != offset + faces.nbytes or not (faces["count"] == 3).all() or not np.isfinite(vertices).all():
        raise ValueError("Invalid body PLY payload")
    return vertices, faces["v"], hashlib.sha256(data[offset:]).hexdigest()


def archive_keys(archive, prefix):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive members")
    keys = set()
    for name in names:
        parts = member_path(name).parts
        if not name.endswith(".pkl"):
            continue
        if len(parts) != 4 or parts[0] != prefix or not parts[2].isdigit() or not Path(parts[3]).stem.isdigit():
            raise ValueError(f"Unexpected annotation path: {name}")
        keys.add("/".join(parts[1:])[:-4])
    return keys


def make_catalog(root, metadata):
    root, metadata = Path(root), Path(metadata)
    provenance = json.loads((metadata / "provenance.json").read_text())
    for record in provenance["files"]:
        if sha256(metadata / record["file"]) != record["sha256"]:
            raise ValueError("Official metadata hash changed")
    with (metadata / "train.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    mapping = {(row["sequence_name"], row["id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Duplicate metadata tracks")
    sources = {}
    keysets = []
    for name in ("train_body", "train_hsc"):
        path = root / "raw/train" / (name + ".zip")
        info = path.stat()
        with zipfile.ZipFile(path) as archive:
            keysets.append(archive_keys(archive, name))
        sources[name] = {"path": str(path.relative_to(root)), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    body, contact = keysets
    if contact - body:
        raise ValueError("Contact annotation has no corresponding body")
    tracks = {}
    for key in sorted(body):
        seq, frame, subject = key.split("/")
        row = mapping[seq, subject]
        track = tracks.setdefault((seq, subject), {"sequence": seq, "subject": subject,
            "capture": row["capture_name"], "scan": row["scan_name"], "gender": row["gender"],
            "camera_ids": [int(v) for v in row["view_id"].split(",")],
            "moving_camera": row["moving_cam"], "frames": [], "body_only_frames": []})
        track["frames" if key in contact else "body_only_frames"].append(int(frame))
    # Connected components expose cross-capture subjects; no train/val split is silently assigned.
    groups = [{t["capture"]} for t in tracks.values()]
    subjects = {}
    for track in tracks.values():
        subjects.setdefault(track["subject"], set()).add(track["capture"])
    groups.extend(subjects.values())
    merged = []
    for group in groups:
        overlap = [other for other in merged if group & other]
        for other in overlap:
            group = group | other; merged.remove(other)
        merged.append(group)
    return {"schema_version": VERSION, "kind": "ground_truth_annotations", "training_ready": False,
            "sources": sources, "metadata": provenance, "tracks": list(tracks.values()),
            "counts": {"sequences": len({x[0] for x in tracks}), "tracks": len(tracks),
                       "body_frames": len(body), "contact_frames": len(contact), "body_only_frames": len(body - contact)},
            "excluded_missing_contact": sorted(body - contact),
            "scene_subject_components": sorted(sorted(group) for group in merged),
            "split_status": "unassigned; components must remain intact for scene+subject isolation",
            "image_alignment": "pending complete JPG archive, filenames, dimensions and K scaling audit"}


def calibration_catalog(root, catalog):
    root = Path(root)
    result = {"coordinate": "calibrated_multicamera", "units": "meters",
              "camera_formula": "p_camera_column = CameraMatrix @ [p_multicam_column; 1]",
              "world_formula": "p_world_row = c * p_multicam_row @ R + t",
              "world_source": "rich_toolkit/multicam2world.py at " + catalog["metadata"]["revision"],
              "intrinsics_resolution": "original calibration; JPG resolution/scale unverified", "captures": {}}
    for track in catalog["tracks"]:
        capture, scan = track["capture"], track["scan"]
        key = capture + "/" + scan
        if key in result["captures"]:
            continue
        base = root / "extracted/scan_calibration" / capture
        scene = base / (scan + ".ply")
        if not scene.is_file():
            raise ValueError(f"Missing official scene: {scene}")
        suffix = "" if scan == "scan_camcoord" else "_" + scan.removeprefix("scan_").removesuffix("_scene_camcoord")
        world = root / "extracted/multicam2world" / (capture + suffix + "_multicam2world.json")
        transform = json.loads(world.read_text())
        rotation = np.asarray(transform["R"]); scale = float(transform["c"])
        translation = np.asarray(transform["t"]).reshape(3)
        # ParkingLot1's official R is rounded to three decimals. Preserve it;
        # do not silently orthogonalize or treat this as a precision audit pass.
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3) or not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
            raise ValueError("Invalid world Sim(3)")
        matrix = np.eye(4); matrix[:3, :3] = scale * rotation.T; matrix[:3, 3] = translation
        cameras = {}
        for path in sorted((base / "calibration").glob("*.xml")):
            xml = ET.parse(path).getroot(); camera = {}
            for tag in ("CameraMatrix", "Intrinsics", "Distortion"):
                node = xml.find(tag)
                value = np.fromstring(node.findtext("data"), sep=" ").reshape(int(node.findtext("rows")), int(node.findtext("cols")))
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite calibration")
                camera[tag] = value.tolist()
            cameras[str(int(path.stem))] = camera
        result["captures"][key] = {"scan_path": str(scene.relative_to(root)), "scan_sha256": sha256(scene),
            "world_source_path": str(world.relative_to(root)), "raw_world_transform": transform,
            "rotation_orthogonality_max_error": float(np.abs(rotation.T @ rotation - np.eye(3)).max()),
            "transform_precision_audit": "pending; raw official coefficients preserved without orthogonalization",
            "world_from_multicam_column_4x4": matrix.tolist(), "cameras": cameras}
    for track in catalog["tracks"]:
        cameras = result["captures"][track["capture"] + "/" + track["scan"]]["cameras"]
        if any(str(camera) not in cameras for camera in track["camera_ids"]):
            raise ValueError("Metadata references missing camera")
    return result


class AnnotationReader:
    def __init__(self, body, contact):
        self.body, self.contact = body, contact
        self.obj_hash = self.ply_hash = None
        self.faces = None

    def read(self, sequence, subject, frame):
        key = f"{sequence}/{frame:05d}/{subject}"
        params = numeric_pickle(self.body.read(f"train_body/{key}.pkl"))
        if set(params) != set(BODY_SHAPES) or any(params[k].shape != (1, n) or params[k].dtype != np.float32 for k, n in BODY_SHAPES.items()):
            raise ValueError(f"Unexpected body parameter schema: {key}")
        label = numeric_pickle(self.contact.read(f"train_hsc/{key}.pkl"))
        if set(label) != {"contact", "s2m_dist_id", "closest_triangles_id"} or label["contact"].shape != (6890,) or label["s2m_dist_id"].shape != (10475, 3) or label["closest_triangles_id"].shape != (10475, 3, 3):
            raise ValueError(f"Unexpected contact schema: {key}")
        if not np.isin(label["contact"], [0, 1]).all():
            raise ValueError("SMPL contact is not binary")
        vertices, faces, ply_hash = ply_mesh(self.body.read(f"train_body/{key}.ply"))
        contact_vertices, contact_mask, contact_valid, obj_hash, obj_faces = obj_mesh(self.contact.read(f"train_hsc/{key}.obj"), self.obj_hash)
        if self.ply_hash is None:
            if not np.array_equal(faces, obj_faces):
                raise ValueError("Body/contact face indices differ")
            self.faces = faces.copy(); self.ply_hash = ply_hash; self.obj_hash = obj_hash
        elif self.ply_hash != ply_hash:
            raise ValueError("Body face topology/order changed")
        error = float(np.abs(vertices - contact_vertices).max())
        if error > 2e-6:
            raise ValueError(f"Body/contact vertex correspondence mismatch: {key}, {error} meters")
        arrays = {"body_" + k: v[0] for k, v in params.items()}
        arrays.update(vertices_multicam=vertices, contact_smplx=contact_mask,
                      contact_smplx_valid=contact_valid, contact_smpl=label["contact"].astype(bool),
                      contact_smpl_valid=np.ones(6890, dtype=bool), s2m_dist_id=label["s2m_dist_id"])
        return arrays, error


def write_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def prepare_annotations(root, metadata, output, sample=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = make_catalog(root, metadata)
    state_path = output / ("sample_audit.json" if sample else "state.json")
    signature = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    state = {"schema_version": VERSION, "catalog_signature": signature, "status": "running",
             "mode": "sample" if sample else "full", "completed_frames": 0, "shards": [], "started": time.time()}
    if state_path.exists() and not sample:
        state = json.loads(state_path.read_text())
        if state["catalog_signature"] != signature or state["schema_version"] != VERSION:
            raise ValueError("Source/catalog changed; choose a new output version")
        state["status"] = "running"
    save_json(output / "catalog.json", catalog)
    save_json(output / "calibration.json", calibration_catalog(root, catalog))
    previous = {item["path"]: item for item in state["shards"]}
    save_json(state_path, state)
    try:
        with zipfile.ZipFile(root / "raw/train/train_body.zip") as body, zipfile.ZipFile(root / "raw/train/train_hsc.zip") as contact:
            reader = AnnotationReader(body, contact)
            first = catalog["tracks"][0]
            reader.read(first["sequence"], first["subject"], first["frames"][0])
            write_npz(output / "smplx_topology.npz", {"faces": reader.faces})
            state["topology_sha256"] = sha256(output / "smplx_topology.npz")
            state["tracks"] = []
            for track in catalog["tracks"]:
                frames = track["frames"]
                if sample:
                    frames = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
                for start in range(0, len(frames), 32):
                    batch = frames[start:start + 32]
                    relative = f"shards/{track['sequence']}/{track['subject']}/{start:06d}.npz"
                    if not sample and relative in previous:
                        if sha256(output / relative) != previous[relative]["sha256"]:
                            raise ValueError(f"Changed completed shard: {relative}")
                        continue
                    records, errors = zip(*(reader.read(track["sequence"], track["subject"], frame) for frame in batch))
                    arrays = {key: np.stack([record[key] for record in records]) for key in records[0]}
                    stats = {"sequence": track["sequence"], "subject": track["subject"], "frames": batch,
                             "vertex_max_abs_error_m": max(errors),
                             "smplx_contact_positive_vertices": int(arrays["contact_smplx"].sum()),
                             "smplx_contact_valid_vertices": int(arrays["contact_smplx_valid"].sum()),
                             "missing_smplx_color_frames": [frame for frame, valid in zip(batch, arrays["contact_smplx_valid"]) if not valid.any()],
                             "smpl_contact_positive_vertices": int(arrays["contact_smpl"].sum())}
                    if sample:
                        state["tracks"].append(stats)
                    else:
                        arrays["frame_ids"] = np.asarray(batch, dtype=np.int64)
                        write_npz(output / relative, arrays)
                        stats.update(path=relative, sha256=sha256(output / relative))
                        state["shards"].append(stats)
                    state["completed_frames"] += len(batch); state["updated"] = time.time()
                    save_json(state_path, state)
                print(f"{track['sequence']}/{track['subject']}: {state['completed_frames']} frames verified", flush=True)
        for source in catalog["sources"].values():
            info = (root / source["path"]).stat()
            if info.st_size != source["bytes"] or info.st_mtime_ns != source["mtime_ns"]:
                raise ValueError("Source archive changed during annotation preparation")
        if not sample and state["completed_frames"] != catalog["counts"]["contact_frames"]:
            raise ValueError("Packed frame count differs from the source catalog")
        state["status"] = "complete"; state["completed"] = time.time()
        state["training_ready"] = False
        state["limitations"] = ["GT annotations only; frozen frontend inputs and independent geometric audit pending",
                                "JPG alignment and resolution pending", "Official validation/test not downloaded",
                                "Uncolored contact OBJ: SMPL-X contact_valid=false; not a negative label",
                                f"{catalog['counts']['body_only_frames']} missing contact frames recorded in catalog; no fabricated labels"]
        save_json(state_path, state)
        return state
    except Exception as error:
        state.update(status="failed", error=str(error), updated=time.time())
        save_json(state_path, state)
        raise
