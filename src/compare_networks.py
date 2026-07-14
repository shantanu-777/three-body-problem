"""
src/compare_networks.py — CouplingFlow vs FlowMatching comparison.

Why this file exists
--------------------
We train the same amortized-posterior setup with two different inference
(generative) networks and ask which one recovers the initial conditions better
and is better calibrated:

  * CouplingFlow  — a discrete normalizing flow (fast to sample)
  * FlowMatching  — a continuous flow trained by flow matching (ODE sampling)

Both are trained separately (see inference.py --inference-network) and saved to
different checkpoint directories. This script loads each *trained* model (no
retraining), samples posteriors on a shared held-out test set in small batches
(to keep memory bounded — FlowMatching's ODE sampler is memory-hungry), and
compares:

  - parameter recovery (Pearson r between posterior mean and truth)
  - calibration (empirical coverage of the 90% credible interval)
  - posterior contraction (posterior std / prior std)
  - sampling speed (wall-clock per 1000 posterior draws)

Outputs
-------
- results/compare_networks.png   — grouped bars: recovery r and coverage per parameter
- results/compare_networks.json  — all metrics + timing for both networks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplcache"))

import keras  # noqa: E402

import config  # noqa: E402
from src.inference import load_dataset, train_val_split  # noqa: E402
from src.observables import (  # noqa: E402
    apply_observable_scales_to_dataset,
    load_observable_scales,
)
from src.priors import PARAMETER_NAMES  # noqa: E402


def _sample_in_batches(approx, observables: np.ndarray, num_samples: int, batch: int) -> np.ndarray:
    """Sample posteriors batch-by-batch to keep memory bounded. Returns (N, S, P)."""
    out = []
    for start in range(0, len(observables), batch):
        chunk = observables[start : start + batch]
        s = approx.sample(num_samples=num_samples, conditions={"observables": chunk})
        key = "parameters" if "parameters" in s else "inference_variables"
        out.append(np.asarray(s[key]))
    return np.concatenate(out, axis=0)


def evaluate_checkpoint(
    checkpoint_dir: Path,
    dataset_path: str | Path,
    n_eval: int,
    num_samples: int,
    batch: int,
    seed: int | None,
) -> dict:
    """Load one trained model and compute recovery / calibration / contraction / timing."""
    data = load_dataset(dataset_path)
    _, test_raw = train_val_split(data, val_fraction=config.TRAIN_VAL_FRACTION, seed=seed)
    scales = load_observable_scales(checkpoint_dir / config.OBSERVABLE_SCALES_FILE)

    n_eval = min(n_eval, len(test_raw["parameters"]))
    truth = test_raw["parameters"][:n_eval]
    obs = apply_observable_scales_to_dataset(
        {"parameters": truth, "observables": test_raw["observables"][:n_eval]}, scales
    )["observables"]

    approx = keras.saving.load_model(str(checkpoint_dir / "three_body_npe.keras"))

    t0 = time.perf_counter()
    post = _sample_in_batches(approx, obs, num_samples=num_samples, batch=batch)
    sample_seconds = time.perf_counter() - t0
    total_draws = n_eval * num_samples

    post_mean = post.mean(axis=1)
    post_std = post.std(axis=1)
    prior_std = np.maximum(data["parameters"].std(axis=0), 1e-12)

    recovery, coverage, contraction = {}, {}, {}
    lo = np.percentile(post, 5.0, axis=1)
    hi = np.percentile(post, 95.0, axis=1)
    for i, name in enumerate(PARAMETER_NAMES):
        recovery[name] = float(np.corrcoef(post_mean[:, i], truth[:, i])[0, 1])
        coverage[name] = float(np.mean((truth[:, i] >= lo[:, i]) & (truth[:, i] <= hi[:, i])))
        contraction[name] = float(np.mean(post_std[:, i]) / prior_std[i])

    return {
        "n_eval": int(n_eval),
        "num_posterior_samples": int(num_samples),
        "recovery_correlation": recovery,
        "mean_recovery_correlation": float(np.mean(list(recovery.values()))),
        "coverage90": coverage,
        "mean_coverage90": float(np.mean(list(coverage.values()))),
        "contraction": contraction,
        "mean_contraction": float(np.mean(list(contraction.values()))),
        "sample_seconds": sample_seconds,
        "seconds_per_1k_draws": float(sample_seconds / total_draws * 1000.0),
    }


def run_comparison(
    dataset_path: str | Path,
    coupling_dir: str | Path,
    flowmatching_dir: str | Path,
    results_dir: str | Path | None = None,
    n_eval: int = 1000,
    num_samples: int = 500,
    batch: int = 200,
    seed: int | None = None,
) -> dict:
    results_dir = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("evaluating CouplingFlow...")
    cf = evaluate_checkpoint(Path(coupling_dir), dataset_path, n_eval, num_samples, batch, seed)
    print(f"  recovery {cf['mean_recovery_correlation']:.3f} | "
          f"coverage {cf['mean_coverage90']:.3f} | "
          f"{cf['seconds_per_1k_draws']:.3f}s / 1k draws")

    print("evaluating FlowMatching...")
    fm = evaluate_checkpoint(Path(flowmatching_dir), dataset_path, n_eval, num_samples, batch, seed)
    print(f"  recovery {fm['mean_recovery_correlation']:.3f} | "
          f"coverage {fm['mean_coverage90']:.3f} | "
          f"{fm['seconds_per_1k_draws']:.3f}s / 1k draws")

    # ---- figure: grouped bars per parameter ----
    names = PARAMETER_NAMES
    x = np.arange(len(names))
    w = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].bar(x - w / 2, [cf["recovery_correlation"][n] for n in names], w,
                label="CouplingFlow", color="#3b5b92")
    axes[0].bar(x + w / 2, [fm["recovery_correlation"][n] for n in names], w,
                label="FlowMatching", color="#c0392b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("recovery correlation  r")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Parameter recovery (higher = better)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].axhline(0.9, ls="--", color="gray", label="nominal 90%")
    axes[1].bar(x - w / 2, [cf["coverage90"][n] for n in names], w,
                label="CouplingFlow", color="#3b5b92")
    axes[1].bar(x + w / 2, [fm["coverage90"][n] for n in names], w,
                label="FlowMatching", color="#c0392b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("coverage of 90% credible interval")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Calibration (closer to 0.9 = better)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"CouplingFlow vs FlowMatching  (3D, {cf['n_eval']} test systems)", fontsize=13
    )
    fig.tight_layout()
    fig_path = results_dir / "compare_networks.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")

    summary = {"coupling": cf, "flowmatching": fm}
    summary_path = results_dir / "compare_networks.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CouplingFlow vs FlowMatching")
    parser.add_argument("--data", type=str, default="data/train_full_3d.npz")
    parser.add_argument("--coupling-dir", type=str, default="checkpoints")
    parser.add_argument("--flowmatching-dir", type=str, default="checkpoints_fm")
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_comparison(
        args.data,
        coupling_dir=args.coupling_dir,
        flowmatching_dir=args.flowmatching_dir,
        results_dir=args.results_dir,
        n_eval=args.n_eval,
        num_samples=args.num_samples,
        batch=args.batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
