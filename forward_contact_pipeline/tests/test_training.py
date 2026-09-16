"""Contract, leakage and continuation regression tests using numeric fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_training_fixture import create_fixture
from build_training_index import build
from contact_streaming.training.components import ContactMetrics, ContactObjective, FusionModel
from contact_streaming.training.config import inside, load_config
from contact_streaming.training.data import NumericCache, ShardShuffleSampler, digest
from contact_streaming.training.engine import evaluate_checkpoint, prepare, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        # All generated work stays on the project NAS.
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="training_test_", dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        create_fixture(self.root / "data")
        paths = self.root / "paths.local.yaml"
        paths.write_text("data_root: data\nrun_root: runs\n")
        self.cfg = load_config(ROOT / "configs/experiments/smoke.yaml", paths)

    def adapter(self):
        return NumericCache(self.root / "data", "index.json")

    def edit_index(self, change):
        p = self.root / "data/index.json"
        data = json.loads(p.read_text()); change(data)
        p.write_text(json.dumps(data))

    def test_paths_and_split_leakage(self):
        with self.assertRaises(ValueError):
            inside(self.root, "../outside")
        self.edit_index(lambda d: d["records"][2].update(subject_id=d["records"][0]["subject_id"]))
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            self.adapter()

    def test_index_builder_and_shard_sampling(self):
        result = build(self.root / "data", self.root / "data/index.json", "new_index.json")
        self.assertEqual(result["shards"], 4)
        self.assertIn("not performed", result["geometry_audit"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            build(self.root / "data", self.root / "data/index.json", "new_index.json")
        adapter = self.adapter(); adapter.audit("smoke")
        data = adapter.dataset("train")
        indices = list(ShardShuffleSampler(data, torch.Generator().manual_seed(123)))
        self.assertEqual(sorted(indices), list(range(len(data))))
        shard_order = [data.items[i][0] for i in indices]
        transitions = sum(a != b for a, b in zip(shard_order, shard_order[1:]))
        self.assertEqual(transitions, 1)

    def test_preflight_rejects_fusion_channel_mismatch(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"] = {"factory": "contact_streaming.training.components:FusionModel", "kwargs": {"hidden": 16, "rgb_dim": 7}}
        with self.assertRaisesRegex(ValueError, "rgb_dim"):
            prepare(cfg, ("train", "val"))

    def test_unapproved_real_data_and_single_class_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Real experiments"):
            self.adapter().audit("experiment")
        for i in (0, 1):
            p = self.root / "data" / f"{i}.targets.npz"
            with np.load(p) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["contact"][:] = 1
            np.savez(p, **arrays)
        with self.assertRaisesRegex(ValueError, "only one contact class"):
            self.adapter().audit("smoke")

    def test_frozen_contract_binds_audit_and_payload_hashes(self):
        def real(d):
            d["provenance"] = {"kind": "real", "input_source": "frozen_backbone", "units": "m", "coordinate_frame": "contact_local",
                               "temporal_context": "current_and_past", "topology": "smplx", "backbone_id": "test-fixture",
                               "backbone_revision": "unit-test", "feature_version": "v1", "label_source": "unit-test"}
            d["audit_report"] = "audit.json"
        self.edit_index(real)
        p = self.root / "data/audit.json"
        report = {"manifest_sha256": "stale", "checks": dict.fromkeys(("coordinates", "topology", "frame_alignment", "label_distribution", "frontend_geometry"), True)}
        p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.adapter().audit("experiment")
        report["manifest_sha256"] = digest(self.root / "data/index.json")
        p.write_text(json.dumps(report))
        self.adapter().audit("experiment")
        with (self.root / "data/0.inputs.npz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since audit"):
            self.adapter().audit("experiment")

    def test_masked_labels_and_optional_fusion(self):
        torch.set_num_threads(1)
        adapter = self.adapter(); adapter.audit("smoke")
        item = adapter.dataset("train")[0]
        inputs = {k: v[None] for k, v in item["inputs"].items()}
        targets = {k: v[None] for k, v in item["targets"].items()}
        net = FusionModel(hidden=16, rgb_dim=6, gaussian_dim=3)
        outputs = net(inputs)
        targets["contact_valid"][..., 0] = 0
        loss = ContactObjective()(outputs, targets)
        changed = copy.deepcopy(targets); changed["contact"][..., 0] = 1 - changed["contact"][..., 0]
        self.assertEqual(loss.item(), ContactObjective()(outputs, changed).item())
        loss.backward()
        metrics = ContactMetrics(); metrics.update(outputs, targets)
        self.assertEqual(metrics.compute()["valid_vertices"], 7)
        inputs["point_valid"].zero_()
        self.assertTrue(torch.isfinite(net(inputs)["contact_logits"]).all())

    def test_resume_matches_uninterrupted_training_and_test_is_explicit(self):
        cfg = copy.deepcopy(self.cfg); cfg["name"] = "full"
        train(cfg)
        cfg["name"] = "resumed"; cfg["trainer"]["epochs"] = 1
        train(cfg)
        folder = self.root / "runs/resumed"
        cfg["trainer"]["epochs"] = 2
        train(cfg, folder / "last.pt")
        full = torch.load(self.root / "runs/full/last.pt", weights_only=True)
        resumed = torch.load(folder / "last.pt", weights_only=True)
        self.assertEqual(full["history"], resumed["history"])
        for key in full["model"]:
            self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)
        self.assertNotIn("test", json.loads((folder / "audit.json").read_text())["splits"])
        metrics = evaluate_checkpoint(cfg, folder / "best.pt", "test")
        self.assertEqual(metrics["split"], "test")
        cfg["objective"]["kwargs"]["positive_weight"] = 2
        with self.assertRaisesRegex(ValueError, "differ"):
            train(cfg, folder / "last.pt")


if __name__ == "__main__":
    unittest.main()
