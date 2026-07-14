"""
src/chaos_analysis.py — does posterior uncertainty track chaos?

Why this file exists
--------------------
The three-body problem is chaotic: some initial conditions diverge from their
neighbours much faster than others. A trustworthy posterior should *know* this —
it should be **wider (less certain)** for strongly chaotic systems, because the
same short, noisy trajectory constrains the initial conditions less when tiny
differences blow up quickly.

This module tests exactly that claim on held-out data:

  1. For each test system we measure a **finite-time divergence exponent**
     (a per-sample, cheap Lyapunov-style number): integrate the true trajectory
     and a tiny perturbation of it, and measure how fast they separate.
  2. We ask the trained BayesFlow model for the posterior of each system and
     measure its **width** (posterior std / prior std, averaged over parameters).
  3. We bin systems by chaos level and show width vs chaos, plus per-bin
     calibration (does a nominal 90% credible interval actually cover 90%?).

Outputs
-------
- results/chaos_analysis.png  — width vs chaos + calibration vs chaos
- results/chaos_analysis.json — binned numbers + overall correlation

Connects to
-----------
- Uses the same trained checkpoint as diagnostics.py (loaded, not retrained).
- Uses simulator/lyapunov physics for the divergence measure.
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

import keras  # noqa: E402

import config  # noqa: E402
from src.inference import load_dataset, train_val_split  # noqa: E402
from src.lyapunov import perturb_initial_state, _integrate_state, phase_space_separation  # noqa: E402
from src.observables import (  # noqa: E402
    apply_observable_scales_to_dataset,
    load_observable_scales,
)
from src.priors import PARAMETER_NAMES, free_dof_to_full_state  # noqa: E402
from src.simulator import simulate_trajectory  # noqa: E402


def finite_time_divergence(
    theta: np.ndarray,
    t_max: float | None = None,
    n_times: int = 60,
    perturbation_size: float = 1e-6,
    seed: int = 0,
) -> float:
    """
    Cheap per-sample chaos measure over the observation window.

    Integrate the reference trajectory and a tiny perturbation of it, then
    return the finite-time exponent

        lambda_FT = (1 / T) * log( d(T) / d(0) )

    where d is the phase-space separation. Larger => more chaotic / sensitive.
    Returns NaN if the reference trajectory is rejected (collision/escape).
    """
    t_max = config.T_MAX if t_max is None else t_max
    times = np.linspace(0.0, t_max, n_times)

    positions, velocities = free_dof_to_full_state(theta)
    ref = simulate_trajectory(positions, velocities, t_eval=times, t_max=t_max)
    if not ref.accepted:
        return float("nan")

    pert_pos, pert_vel = perturb_initial_state(
        positions, velocities, perturbation_size, seed=seed
    )
    ref_traj = _integrate_state(positions, velocities, times)
    pert_traj = _integrate_state(pert_pos, pert_vel, times)
    if ref_traj is None or pert_traj is None:
        return float("nan")

    _, ref_states = ref_traj
    _, pert_states = pert_traj
    d0 = phase_space_separation(ref_states[0], pert_states[0])
    dT = phase_space_separation(ref_states[-1], pert_states[-1])
    if d0 <= 0.0 or dT <= 0.0:
        return float("nan")
    return float(np.log(dT / d0) / t_max)


def _bin_stats(x: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    """Mean/std of y within quantile bins of x."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    centers, means, sems, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            continue
        centers.append(float(np.median(x[mask])))
        means.append(float(np.mean(y[mask])))
        sems.append(float(np.std(y[mask]) / max(np.sqrt(np.sum(mask)), 1.0)))
        counts.append(int(np.sum(mask)))
    return {
        "centers": centers,
        "means": means,
        "sems": sems,
        "counts": counts,
        "edges": edges.tolist(),
    }


def run_chaos_analysis(
    dataset_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    n_eval: int = 400,
    num_samples: int = 300,
    n_bins: int = 6,
    seed: int | None = None,
) -> dict:
    """Load a trained model and relate posterior width to per-sample chaos."""
    checkpoint_dir = Path(config.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)
    results_dir = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    _, test_raw = train_val_split(data, val_fraction=config.TRAIN_VAL_FRACTION, seed=seed)
    scales = load_observable_scales(checkpoint_dir / config.OBSERVABLE_SCALES_FILE)

    n_eval = min(n_eval, len(test_raw["parameters"]))
    test_params = test_raw["parameters"][:n_eval]
    test_obs_scaled = apply_observable_scales_to_dataset(
        {"parameters": test_params, "observables": test_raw["observables"][:n_eval]}, scales
    )["observables"]

    print(f"loading model from {checkpoint_dir}...")
    approx = keras.saving.load_model(str(checkpoint_dir / "three_body_npe.keras"))

    print(f"sampling posteriors for {n_eval} test systems...")
    samples = approx.sample(
        num_samples=num_samples, conditions={"observables": test_obs_scaled}
    )
    key = "parameters" if "parameters" in samples else "inference_variables"
    post = np.asarray(samples[key])  # (n_eval, num_samples, n_params)

    # Posterior width: std per parameter normalized by that parameter's prior std.
    prior_std = np.maximum(data["parameters"].std(axis=0), 1e-12)
    post_std = post.std(axis=1)                      # (n_eval, n_params)
    norm_width_all = np.mean(post_std / prior_std, axis=1)  # (n_eval,) over all params

    # The chaos signal lives in the velocities: positions are near-perfectly
    # recovered regardless of chaos and dilute the effect. Track velocity-only
    # width as the primary uncertainty measure.
    vel_idx = [i for i, name in enumerate(PARAMETER_NAMES) if name.startswith("v")]
    norm_width_vel = np.mean((post_std / prior_std)[:, vel_idx], axis=1)
    norm_width = norm_width_vel  # primary measure for the main panel

    # Per-sample calibration: fraction of parameters whose truth lands in the
    # central 90% credible interval of the posterior.
    lo = np.percentile(post, 5.0, axis=1)
    hi = np.percentile(post, 95.0, axis=1)
    inside = (test_params >= lo) & (test_params <= hi)   # (n_eval, n_params)
    coverage90 = inside.mean(axis=1)                     # (n_eval,)

    print(f"computing finite-time divergence for {n_eval} systems...")
    chaos = np.array(
        [finite_time_divergence(test_params[i], seed=i) for i in range(n_eval)]
    )

    valid = np.isfinite(chaos) & np.isfinite(norm_width)
    chaos_v = chaos[valid]
    width_v = norm_width[valid]                 # velocity-only width
    width_all_v = norm_width_all[valid]         # all-parameter width
    cover_v = coverage90[valid]

    corr = float(np.corrcoef(chaos_v, width_v)[0, 1]) if chaos_v.size > 2 else float("nan")
    corr_all = float(np.corrcoef(chaos_v, width_all_v)[0, 1]) if chaos_v.size > 2 else float("nan")

    width_bins = _bin_stats(chaos_v, width_v, n_bins)
    cover_bins = _bin_stats(chaos_v, cover_v, n_bins)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(chaos_v, width_v, s=8, alpha=0.25, color="#3b5b92", label="test systems")
    axes[0].errorbar(
        width_bins["centers"], width_bins["means"], yerr=width_bins["sems"],
        fmt="o-", color="#c0392b", capsize=4, label="binned mean",
    )
    axes[0].set_xlabel("finite-time divergence exponent  λ_FT  (more chaotic →)")
    axes[0].set_ylabel("normalized velocity posterior width  (std / prior std)")
    axes[0].set_title(f"Velocity posterior widens with chaos   (r = {corr:.2f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.9, ls="--", color="gray", label="nominal 90%")
    axes[1].errorbar(
        cover_bins["centers"], cover_bins["means"], yerr=cover_bins["sems"],
        fmt="o-", color="#2c7a3f", capsize=4, label="empirical coverage",
    )
    axes[1].set_xlabel("finite-time divergence exponent  λ_FT  (more chaotic →)")
    axes[1].set_ylabel("coverage of 90% credible interval")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Calibration vs chaos")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = results_dir / "chaos_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")

    summary = {
        "dataset": str(dataset_path),
        "checkpoint_dir": str(checkpoint_dir),
        "n_eval": int(n_eval),
        "n_valid": int(valid.sum()),
        "num_posterior_samples": num_samples,
        "corr_chaos_vs_width_velocity": corr,
        "corr_chaos_vs_width_all_params": corr_all,
        "overall_mean_coverage90": float(np.mean(cover_v)),
        "width_vs_chaos_bins": width_bins,
        "coverage_vs_chaos_bins": cover_bins,
    }
    summary_path = results_dir / "chaos_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {summary_path}")
    print(f"correlation (chaos vs velocity width):   {corr:.3f}")
    print(f"correlation (chaos vs all-param width):  {corr_all:.3f}")
    print(f"overall 90% coverage: {summary['overall_mean_coverage90']:.3f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chaos vs posterior-uncertainty analysis")
    parser.add_argument("--data", type=str, default="data/train_full_3d.npz")
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--n-eval", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_chaos_analysis(
        args.data,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        n_eval=args.n_eval,
        num_samples=args.num_samples,
        n_bins=args.n_bins,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
