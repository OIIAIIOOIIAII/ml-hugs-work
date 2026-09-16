from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contact_streaming.alignment import robust_sim3
from contact_streaming.models import build_model
from contact_streaming.packet import ContactPacket
from contact_streaming.surface_proxy import fit_local_plane


class CoreTests(unittest.TestCase):
    def test_robust_sim3(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(20, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        dst = 1.7 * (src @ rotation.T) + np.array([1, 2, 3])
        result = robust_sim3(src, dst)
        self.assertLess(np.max(np.abs(result.transform(src) - dst)), 1e-4)

    def test_surface_plane(self):
        x, z = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.2, 0.2, 20))
        points = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=1)
        result = fit_local_plane(np.array([0, 0.05, 0]), points, up_axis=1)
        self.assertAlmostEqual(result.signed_distance, 0.05, places=3)
        self.assertGreater(result.confidence, 0.9)

    def test_packet_roundtrip(self):
        packet = ContactPacket(3, 1000, 1, 0, np.zeros(6), np.zeros(12), np.ones(2), np.zeros(2))
        payload = packet.encode()
        decoded = ContactPacket.decode(payload)
        self.assertEqual(len(payload), 62)
        self.assertEqual(decoded.frame_id, 3)

    def test_models(self):
        for name in ["gru", "tcn"]:
            model = build_model(name, 58, hidden_dim=16, layers=2)
            output = model(torch.zeros(4, 8, 58))
            self.assertEqual(tuple(output.contact_logits.shape), (4, 2))
            self.assertEqual(tuple(output.pose_residual.shape), (4, 12))


if __name__ == "__main__":
    unittest.main()
