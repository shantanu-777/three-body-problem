"""
src/diagnostics.py — evaluate the trained BayesFlow posterior.

Why this file exists
--------------------
Training loss going down is not enough. The project spec requires:

  * Simulation-Based Calibration (SBC) — is the posterior honest about its uncertainty?
  * Posterior contraction — is the posterior narrower than the prior?
  * Parameter recovery — do true parameters fall near posterior mass?
  * Posterior predictive checks — do simulations from posterior samples look like data?

BayesFlow provides many of these via ``plot_default_diagnostics``; this module
wraps that and saves figures for the report and slides.

Inputs
------
- Trained workflow (from ``inference.py``)
- Held-out test data (.npz)

Outputs
------
- Diagnostic figures saved under ``results/``
- Summary metrics printed to stdout

Connects to
-----------
- Uses the same data format as ``generate_data.py`` and ``inference.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplcache"))

import config
from src.inference import load_dataset, train_val_split
from src.priors import PARAMETER_NAMES


def _save_figures(figures: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def compute_posterior_contraction(
    posterior_samples: np.ndarray,
    prior_samples: np.ndarray,
) -> dict[str, float]:
    """
    Compare posterior vs prior standard deviation per parameter (contraction).

    Values < 1 mean the posterior is narrower than the prior on that axis.
    """
    post_std = posterior_samples.std(axis=0)
    prior_std = prior_samples.std(axis=0)
    ratio = post_std / np.maximum(prior_std, 1e-12)
    return {
        name: float(r)
        for name, r in zip(PARAMETER_NAMES, ratio)
    }


def run_diagnostics(
    dataset_path: str | Path,
    results_dir: str | Path | None = None,
    num_samples: int | None = None,
    epochs: int | None = None,
    checkpoint_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict:
    """
    Train the workflow, then run the full diagnostic suite on held-out data.
    """
    results_dir = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    num_samples = config.TRAIN_NUM_DIAG_SAMPLES if num_samples is None else num_samples
    checkpoint_dir = Path(config.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)

    from src.inference import train_workflow

    print("training workflow...")
    workflow, _, _ = train_workflow(
        dataset_path,
        epochs=epochs or config.TRAIN_EPOCHS,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    data = load_dataset(dataset_path)
    _, test_data = train_val_split(data, val_fraction=config.TRAIN_VAL_FRACTION, seed=seed)

    print("generating BayesFlow default diagnostics...")
    figures = workflow.plot_default_diagnostics(
        test_data,
        num_samples=num_samples,
        variable_names=PARAMETER_NAMES,
    )
    _save_figures(figures, results_dir)

    print("sampling posteriors for contraction metrics...")
    samples = workflow.sample(
        num_samples=num_samples,
        conditions={"observables": test_data["observables"][: min(50, len(test_data["observables"]))]},
    )
    key = "parameters" if "parameters" in samples else "inference_variables"
    post_samples = samples[key]

    # Flatten across test observations for a global contraction summary.
    post_flat = post_samples.reshape(-1, post_samples.shape[-1])
    prior_flat = data["parameters"]

    contraction = compute_posterior_contraction(post_flat, prior_flat)
    summary = {
        "n_test": len(test_data["parameters"]),
        "num_posterior_samples": num_samples,
        "posterior_contraction_ratio": contraction,
        "mean_contraction_ratio": float(np.mean(list(contraction.values()))),
    }

    summary_path = results_dir / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {summary_path}")
    print(f"mean contraction ratio: {summary['mean_contraction_ratio']:.3f}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BayesFlow posterior diagnostics")
    parser.add_argument("--data", type=str, default="data/train_small.npz")
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--num-samples", type=int, default=config.TRAIN_NUM_DIAG_SAMPLES)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_diagnostics(
        args.data,
        results_dir=args.results_dir,
        num_samples=args.num_samples,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
