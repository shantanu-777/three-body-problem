"""
src/observables.py — separate position/velocity scaling for BayesFlow inputs.

Why this file exists
--------------------
Raw trajectories concatenate positions (scale ~2) and velocities (scale ~0.15).
With equal Gaussian noise sigma_x = sigma_v, velocities are ~13x noisier in
relative terms. Transformers then under-weight velocity channels.

We z-score position and velocity blocks separately (fit on training data only)
and optionally boost velocity channels so the summary network pays attention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class ObservableScales:
    """Per-feature scales for the 12-dimensional observation vector."""

    pos_mean: np.ndarray   # shape (6,)
    pos_std: np.ndarray    # shape (6,)
    vel_mean: np.ndarray   # shape (6,)
    vel_std: np.ndarray    # shape (6,)
    vel_emphasis: float

    def to_dict(self) -> dict:
        return {
            "pos_mean": self.pos_mean.tolist(),
            "pos_std": self.pos_std.tolist(),
            "vel_mean": self.vel_mean.tolist(),
            "vel_std": self.vel_std.tolist(),
            "vel_emphasis": self.vel_emphasis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObservableScales:
        return cls(
            pos_mean=np.asarray(d["pos_mean"], dtype=np.float32),
            pos_std=np.asarray(d["pos_std"], dtype=np.float32),
            vel_mean=np.asarray(d["vel_mean"], dtype=np.float32),
            vel_std=np.asarray(d["vel_std"], dtype=np.float32),
            vel_emphasis=float(d["vel_emphasis"]),
        )


def fit_observable_scales(
    observables: np.ndarray,
    vel_emphasis: float | None = None,
) -> ObservableScales:
    """
    Estimate mean/std for position and velocity blocks from training trajectories.

    observables shape: (N, T, 12)
    """
    vel_emphasis = config.VEL_OBS_EMPHASIS if vel_emphasis is None else vel_emphasis
    pos = observables[..., : config.OBS_N_POS].reshape(-1, config.OBS_N_POS)
    vel = observables[..., config.OBS_N_POS :].reshape(-1, config.OBS_N_VEL)

    return ObservableScales(
        pos_mean=pos.mean(axis=0).astype(np.float32),
        pos_std=np.maximum(pos.std(axis=0), 1e-6).astype(np.float32),
        vel_mean=vel.mean(axis=0).astype(np.float32),
        vel_std=np.maximum(vel.std(axis=0), 1e-6).astype(np.float32),
        vel_emphasis=float(vel_emphasis),
    )


def transform_observables(
    observables: np.ndarray,
    scales: ObservableScales,
) -> np.ndarray:
    """Apply block-wise standardization + velocity emphasis."""
    out = np.asarray(observables, dtype=np.float32).copy()
    out[..., : config.OBS_N_POS] = (
        (out[..., : config.OBS_N_POS] - scales.pos_mean) / scales.pos_std
    )
    out[..., config.OBS_N_POS :] = (
        (out[..., config.OBS_N_POS :] - scales.vel_mean) / scales.vel_std
    ) * scales.vel_emphasis
    return out


def apply_observable_scales_to_dataset(
    data: dict[str, np.ndarray],
    scales: ObservableScales,
) -> dict[str, np.ndarray]:
    """Return a copy of the dataset with scaled observables."""
    return {
        "parameters": data["parameters"],
        "observables": transform_observables(data["observables"], scales),
    }


def save_observable_scales(scales: ObservableScales, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scales.to_dict(), indent=2))


def load_observable_scales(path: str | Path) -> ObservableScales:
    return ObservableScales.from_dict(json.loads(Path(path).read_text()))
