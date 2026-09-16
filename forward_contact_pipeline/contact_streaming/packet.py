from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


MAGIC = b"HCT1"
HEADER = struct.Struct("<4sIQBB")


@dataclass
class ContactPacket:
    frame_id: int
    timestamp_us: int
    contact_mask: int
    flags: int
    root_residual: np.ndarray
    pose_residual: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray

    FLAG_KEYFRAME = 1
    FLAG_RESET = 2
    FLAG_UNCERTAIN = 4

    def encode(self) -> bytes:
        arrays = np.concatenate(
            [self.root_residual.reshape(-1), self.pose_residual.reshape(-1), self.confidence.reshape(-1), self.uncertainty.reshape(-1)]
        ).astype("<f2")
        return HEADER.pack(MAGIC, int(self.frame_id), int(self.timestamp_us), int(self.contact_mask), int(self.flags)) + arrays.tobytes()

    @classmethod
    def decode(cls, payload: bytes) -> "ContactPacket":
        magic, frame_id, timestamp_us, contact_mask, flags = HEADER.unpack_from(payload)
        if magic != MAGIC:
            raise ValueError("Invalid contact packet magic")
        values = np.frombuffer(payload, dtype="<f2", offset=HEADER.size).astype(np.float32)
        if len(values) != 22:
            raise ValueError(f"Expected 22 fp16 values, got {len(values)}")
        return cls(frame_id, timestamp_us, contact_mask, flags, values[:6], values[6:18], values[18:20], values[20:22])


def packet_from_prediction(
    frame_id: int,
    timestamp_sec: float,
    contact_probability: np.ndarray,
    root_residual: np.ndarray,
    pose_residual: np.ndarray,
    uncertainty: np.ndarray,
    mode: str = "Tracking",
    keyframe: bool = False,
) -> ContactPacket:
    contact = np.asarray(contact_probability) >= 0.5
    mask = int(contact[0]) | (int(contact[1]) << 1)
    flags = ContactPacket.FLAG_KEYFRAME if keyframe else 0
    if mode == "Reset":
        flags |= ContactPacket.FLAG_RESET
    elif mode == "Uncertain":
        flags |= ContactPacket.FLAG_UNCERTAIN
    return ContactPacket(
        frame_id=frame_id, timestamp_us=round(timestamp_sec * 1_000_000), contact_mask=mask, flags=flags,
        root_residual=np.asarray(root_residual, np.float32), pose_residual=np.asarray(pose_residual, np.float32),
        confidence=np.asarray(contact_probability, np.float32), uncertainty=np.asarray(uncertainty, np.float32),
    )
