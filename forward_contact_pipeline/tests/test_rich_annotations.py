"""Numeric contracts that prevent mislabeled or misaligned RICH supervision."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_streaming.rich_annotations import AnnotationReader, BODY_SHAPES, calibration_catalog, make_catalog, numeric_pickle, prepare_annotations


def archives(*, shift=0.0, color="0 1 0", permute_face=False, uncolored=False):
    vertices = np.zeros((10475, 3), dtype="<f4")
    vertices[:, 0] = np.arange(10475) * .0001
    faces = np.empty(20908, dtype=[("count", "u1"), ("v", "<i4", (3,))])
    faces["count"] = 3; faces["v"] = [0, 1, 2]
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 10475\nproperty float x\nproperty float y\nproperty float z\nelement face 20908\nproperty list uchar int vertex_indices\nend_header\n"
    obj = "".join(f"v {float(v[0]) + shift:.9f} 0 0 {'' if uncolored else color if i == 0 else '0.4 0.4 0.4'}\n" for i, v in enumerate(vertices))
    obj += ("f 2//2 1//1 3//3\n" if permute_face else "f 1//1 2//2 3//3\n") * 20908
    params = {key: np.zeros((1, n), dtype=np.float32) for key, n in BODY_SHAPES.items()}
    labels = {"contact": np.zeros(6890), "s2m_dist_id": np.zeros((10475, 3)),
              "closest_triangles_id": np.zeros((10475, 3, 3))}
    bodies, contacts = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(bodies, "w") as archive:
        archive.writestr("train_body/seq/00001/001.pkl", pickle.dumps(params, protocol=2))
        archive.writestr("train_body/seq/00001/001.ply", header + vertices.tobytes() + faces.tobytes())
    with zipfile.ZipFile(contacts, "w") as archive:
        archive.writestr("train_hsc/seq/00001/001.pkl", pickle.dumps(labels, protocol=2))
        archive.writestr("train_hsc/seq/00001/001.obj", obj)
    return zipfile.ZipFile(bodies), zipfile.ZipFile(contacts)


class RichAnnotationTests(unittest.TestCase):
    def test_smpl_and_smplx_labels_remain_distinct(self):
        body, contact = archives()
        with body, contact:
            arrays, error = AnnotationReader(body, contact).read("seq", "001", 1)
        self.assertEqual(arrays["contact_smpl"].shape, (6890,))
        self.assertEqual(arrays["contact_smplx"].shape, (10475,))
        self.assertEqual(arrays["contact_smplx"].sum(), 1)
        self.assertEqual(arrays["contact_smpl"].sum(), 0)
        self.assertLess(error, 1e-8)

    def test_absent_vertex_colors_are_unknown_not_negative(self):
        body, contact = archives(uncolored=True)
        with body, contact:
            arrays, _ = AnnotationReader(body, contact).read("seq", "001", 1)
        self.assertFalse(arrays["contact_smplx_valid"].any())
        self.assertTrue(arrays["contact_smpl_valid"].all())

    def test_coordinate_shift_is_rejected(self):
        body, contact = archives(shift=.01)
        with body, contact, self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            AnnotationReader(body, contact).read("seq", "001", 1)

    def test_topology_and_unknown_colors_are_rejected(self):
        for options, message in [({"permute_face": True}, "face indices"), ({"color": "1 0 0"}, "Unknown official")]:
            body, contact = archives(**options)
            with body, contact, self.assertRaisesRegex(ValueError, message):
                AnnotationReader(body, contact).read("seq", "001", 1)

    def test_non_numeric_pickle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-numeric pickle global"):
            numeric_pickle(pickle.dumps(SimpleNamespace(value=1)))

    def test_world_matrix_preserves_official_row_vector_convention(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = Path(temporary); base = root / "extracted/scan_calibration/A"; (base / "calibration").mkdir(parents=True)
            (base / "scan_camcoord.ply").write_bytes(b"numeric scene placeholder")
            camera = "<opencv_storage>"
            for key, array in [("CameraMatrix", np.eye(3, 4)), ("Intrinsics", np.eye(3)), ("Distortion", np.zeros((8, 1)))]:
                camera += f"<{key}><rows>{array.shape[0]}</rows><cols>{array.shape[1]}</cols><data>{' '.join(map(str, array.ravel()))}</data></{key}>"
            (base / "calibration/000.xml").write_text(camera + "</opencv_storage>")
            world = root / "extracted/multicam2world"; world.mkdir()
            transform = {"R": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "c": 2, "t": [1, 2, 3]}
            (world / "A_multicam2world.json").write_text(json.dumps(transform))
            catalog = {"metadata": {"revision": "fixture"}, "tracks": [{"capture": "A", "scan": "scan_camcoord", "camera_ids": [0]}]}
            result = calibration_catalog(root, catalog)["captures"]["A/scan_camcoord"]
            matrix = np.array(result["world_from_multicam_column_4x4"])
            np.testing.assert_allclose(matrix @ [1, 0, 0, 1], [1, 0, 3, 1])
            self.assertEqual(result["raw_world_transform"], transform)

    def test_packed_output_resumes_and_detects_modified_shards(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = Path(temporary); raw = root / "raw/train"; raw.mkdir(parents=True)
            body, contact = archives()
            with body, contact:
                (raw / "train_body.zip").write_bytes(body.fp.getvalue())
                (raw / "train_hsc.zip").write_bytes(contact.fp.getvalue())
            metadata = root / "metadata"; metadata.mkdir()
            (metadata / "train.tsv").write_text("sequence_name\tcapture_name\tscan_name\tid\tmoving_cam\tgender\tview_id\nseq\tA\tscan_camcoord\t001\tX\tmale\t0\n")
            (metadata / "provenance.json").write_text(json.dumps({"files": []}))
            output = root / "out"
            with mock.patch("contact_streaming.rich_annotations.calibration_catalog", return_value={}):
                first = prepare_annotations(root, metadata, output)
                self.assertEqual(first["completed_frames"], 1)
                path = output / first["shards"][0]["path"]
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data["contact_smplx"].shape, (1, 10475))
                    self.assertEqual(data["frame_ids"].tolist(), [1])
                second = prepare_annotations(root, metadata, output)
                self.assertEqual(first["shards"], second["shards"])
                path.write_bytes(b"damaged")
                with self.assertRaisesRegex(ValueError, "Changed completed shard"):
                    prepare_annotations(root, metadata, output)

    def test_missing_labels_and_cross_scene_subject_components(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            root = Path(temporary); raw = root / "raw/train"; raw.mkdir(parents=True)
            metadata = root / "metadata"; metadata.mkdir()
            text = "sequence_name\tcapture_name\tscan_name\tid\tmoving_cam\tgender\tview_id\n"
            text += "a\tA\tscan_camcoord\t001\tX\tmale\t0\nb\tB\tscan_camcoord\t001\tX\tmale\t0\nc\tC\tscan_camcoord\t002\tX\tfemale\t0\n"
            (metadata / "train.tsv").write_text(text)
            (metadata / "provenance.json").write_text(json.dumps({"files": [{"file": "train.tsv", "sha256": hashlib.sha256(text.encode()).hexdigest()}]}))
            keys = ["a/00001/001", "b/00001/001", "c/00001/002"]
            for name in ["train_body", "train_hsc"]:
                with zipfile.ZipFile(raw / (name + ".zip"), "w") as archive:
                    for key in keys + (["a/00002/001"] if name == "train_body" else []):
                        archive.writestr(name + "/" + key + ".pkl", b"unused")
            catalog = make_catalog(root, metadata)
            self.assertEqual(catalog["excluded_missing_contact"], ["a/00002/001"])
            self.assertEqual(catalog["scene_subject_components"], [["A", "B"], ["C"]])
            self.assertFalse(catalog["training_ready"])


if __name__ == "__main__":
    unittest.main()
