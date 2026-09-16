from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RuleConfig:
    enter_distance: float = 0.035
    maintain_distance: float = 0.060
    exit_distance: float = 0.090
    enter_speed: float = 0.10
    maintain_speed: float = 0.16
    min_confidence: float = 0.25
    enter_frames: int = 2
    exit_frames: int = 3
    max_root_correction: float = 0.08
    correction_gain: float = 0.8


class ContactStateMachine:
    """Per-foot hysteresis with Tracking, Uncertain and Reset modes."""

    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()
        self.contact = np.zeros(2, dtype=bool)
        self.enter_count = np.zeros(2, dtype=np.int32)
        self.exit_count = np.zeros(2, dtype=np.int32)
        self.mode = "Tracking"

    def reset(self) -> None:
        self.__init__(self.config)

    def update(
        self,
        distance: np.ndarray,
        speed: np.ndarray,
        confidence: np.ndarray,
        alignment_confidence: float,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        cfg = self.config
        distance = np.abs(np.asarray(distance))
        speed = np.asarray(speed)
        confidence = np.asarray(confidence) * float(alignment_confidence)
        if alignment_confidence <= 0.05 or not np.isfinite(distance).all():
            self.mode = "Reset"
            self.contact[:] = False
            return self.contact.copy(), np.zeros(2, dtype=np.float32), self.mode
        if np.any(confidence < cfg.min_confidence):
            self.mode = "Uncertain"
        else:
            self.mode = "Tracking"
        probability = np.exp(-distance / max(cfg.maintain_distance, 1e-6)) * np.exp(-speed / max(cfg.maintain_speed, 1e-6))
        probability *= np.clip(confidence, 0.0, 1.0)
        for foot in range(2):
            if self.contact[foot]:
                should_exit = distance[foot] > cfg.exit_distance or speed[foot] > cfg.maintain_speed * 1.5
                self.exit_count[foot] = self.exit_count[foot] + 1 if should_exit else 0
                if self.exit_count[foot] >= cfg.exit_frames:
                    self.contact[foot] = False
                    self.exit_count[foot] = 0
            else:
                should_enter = (
                    distance[foot] < cfg.enter_distance
                    and speed[foot] < cfg.enter_speed
                    and confidence[foot] >= cfg.min_confidence
                )
                self.enter_count[foot] = self.enter_count[foot] + 1 if should_enter else 0
                if self.enter_count[foot] >= cfg.enter_frames:
                    self.contact[foot] = True
                    self.enter_count[foot] = 0
        return self.contact.copy(), probability.astype(np.float32), self.mode


def geometric_correction(
    contact: np.ndarray,
    signed_distance: np.ndarray,
    surface_normal: np.ndarray,
    confidence: np.ndarray,
    config: RuleConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return root SE(3), 12D foot-pose residual and projected foot targets."""
    cfg = config or RuleConfig()
    contact = np.asarray(contact, dtype=bool)
    distance = np.asarray(signed_distance, dtype=np.float32)
    normal = np.asarray(surface_normal, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    root = np.zeros(6, dtype=np.float32)
    pose = np.zeros(12, dtype=np.float32)
    targets = -distance[:, None] * normal
    active = np.where(contact & np.isfinite(distance) & (distance < 0.0))[0]
    if len(active):
        correction = (targets[active] * confidence[active, None]).sum(0) / max(confidence[active].sum(), 1e-6)
        norm = np.linalg.norm(correction)
        if norm > cfg.max_root_correction:
            correction *= cfg.max_root_correction / norm
        root[:3] = cfg.correction_gain * correction
        remaining = targets - root[:3]
        # Per-foot xyz targets occupy the first 6 pose slots; rotations remain zero.
        pose[:6] = remaining.reshape(-1)
    return root, pose, targets.astype(np.float32)
