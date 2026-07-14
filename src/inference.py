"""
src/inference.py — train the BayesFlow amortized posterior approximator.

Why this file exists
--------------------
This is Phase 4: we train a **Neural Posterior Estimation (NPE)** model that
learns the inverse map

    noisy trajectory  -->  posterior over initial conditions

BayesFlow combines:
  * a **summary network** (compresses the time series into a fixed-size vector)
  * an **inference network** (a normalizing flow that generates posterior samples)

Inputs
------
- Offline dataset (.npz) with ``parameters`` and ``observables`` arrays

Outputs
------
- Trained workflow saved under ``checkpoints/``
- Training history (loss curves)

Connects to
-----------
- ``diagnostics.py``: evaluates calibration and recovery on a held-out test set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplcache"))

import bayesflow as bf

import config
from src.observables import (
    apply_observable_scales_to_dataset,
    fit_observable_scales,
    save_observable_scales,
)
from src.priors import PARAMETER_NAMES


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load parameters and observables from a .npz file."""
    data = np.load(path)
    return {
        "parameters": data["parameters"].astype(np.float32),
        "observables": data["observables"].astype(np.float32),
    }


def train_val_split(
    data: dict[str, np.ndarray],
    val_fraction: float,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Random train/validation split."""
    n = len(data["parameters"])
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


def prepare_training_data(
    data: dict[str, np.ndarray],
    val_fraction: float,
    seed: int | None,
    checkpoint_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Split data, fit observable scales on train only, apply separate pos/vel scaling.
    """
    train_raw, val_raw = train_val_split(data, val_fraction, seed=seed)
    scales = fit_observable_scales(train_raw["observables"])
    save_observable_scales(scales, checkpoint_dir / config.OBSERVABLE_SCALES_FILE)

    train_data = apply_observable_scales_to_dataset(train_raw, scales)
    val_data = apply_observable_scales_to_dataset(val_raw, scales)
    return train_data, val_data


def build_adapter() -> bf.adapters.Adapter:
    """
    Configure how raw arrays are fed into the networks.

    - ``observables`` is marked as a time series (shape: batch, time, features).
    - Parameters and trajectories are concatenated into the keys BayesFlow expects.
    """
    return (
        bf.adapters.Adapter()
        .convert_dtype(from_dtype="float64", to_dtype="float32")
        .as_time_series("observables")
        .concatenate("parameters", into="inference_variables")
        .concatenate("observables", into="summary_variables")
    )


def build_workflow(checkpoint_dir: str | Path | None = None) -> bf.BasicWorkflow:
    """Create the BayesFlow workflow (summary net + coupling flow)."""
    checkpoint_dir = config.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return bf.BasicWorkflow(
        adapter=build_adapter(),
        inference_network=bf.networks.CouplingFlow(depth=config.INFERENCE_FLOW_DEPTH),
        summary_network=bf.networks.TimeSeriesTransformer(
            summary_dim=config.SUMMARY_DIM,
            embed_dims=config.SUMMARY_EMBED_DIMS,
            num_heads=config.SUMMARY_NUM_HEADS,
        ),
        inference_variables="inference_variables",
        inference_conditions="summary_variables",
        summary_variables="summary_variables",
        standardize=["inference_variables", "summary_variables"],
        initial_learning_rate=config.LEARNING_RATE,
        checkpoint_filepath=str(checkpoint_dir),
        checkpoint_name="three_body_npe",
    )


def train_workflow(
    dataset_path: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    val_fraction: float | None = None,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[bf.BasicWorkflow, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load data, build workflow, and train offline."""
    epochs = config.TRAIN_EPOCHS if epochs is None else epochs
    batch_size = config.TRAIN_BATCH_SIZE if batch_size is None else batch_size
    val_fraction = config.TRAIN_VAL_FRACTION if val_fraction is None else val_fraction
    checkpoint_dir = Path(config.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    train_data, val_data = prepare_training_data(
        data, val_fraction, seed=seed, checkpoint_dir=checkpoint_dir
    )

    workflow = build_workflow(checkpoint_dir=checkpoint_dir)
    history = workflow.fit_offline(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data,
    )

    meta_path = checkpoint_dir / "training_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "epochs": epochs,
                "batch_size": batch_size,
                "n_train": len(train_data["parameters"]),
                "n_val": len(val_data["parameters"]),
                "parameter_names": PARAMETER_NAMES,
                "vel_obs_emphasis": config.VEL_OBS_EMPHASIS,
                "summary_dim": config.SUMMARY_DIM,
                "flow_depth": config.INFERENCE_FLOW_DEPTH,
                "final_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
            },
            indent=2,
        )
    )

    return workflow, train_data, val_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BayesFlow NPE on 3-body data")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train_small.npz",
        help="path to training .npz file",
    )
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Training on {args.data} for {args.epochs} epochs")
    workflow, train_data, val_data = train_workflow(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    print(f"train size: {len(train_data['parameters'])}, val size: {len(val_data['parameters'])}")
    print(f"checkpoints saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
