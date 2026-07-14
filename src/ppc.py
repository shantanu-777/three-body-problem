"""
src/ppc.py — posterior predictive checks (PPC) for the 3-body NPE.

Why this file exists
--------------------
Recovery and calibration tell us the posterior over *parameters* is good. A
posterior predictive check asks a complementary question in *data space*:

    if we take initial conditions sampled from the posterior and simulate them
    forward, do the resulting trajectories reproduce the observed data?

For a trustworthy model the observed (noisy) trajectory should sit inside the
spread of trajectories generated from posterior samples. This is exactly the
"posterior predictive checks" item in the project brief.

Outputs
-------
- results/ppc_trajectories.png — for several test systems: posterior-predictive
  trajectories (thin lines) vs the true trajectory (thick) and observed points.
- results/ppc_summary.json — posterior-predictive coverage of observed data.
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
from src.inference import load_dataset  # noqa: E402
from src.lyapunov import _integrate_state  # noqa: E402
from src.observables import apply_observable_scales_to_dataset, load_observable_scales  # noqa: E402
from src.priors import free_dof_to_full_state, trajectory_to_observation_array  # noqa: E402
from src.simulator import simulate_trajectory, unpack_state  # noqa: E402

_BODY_COLORS = ("#c0392b", "#2c7a3f", "#3b5b92")


def _dense_trajectory(theta: np.ndarray, n: int = 200) -> np.ndarray | None:
    """Integrate initial conditions on a dense grid; return positions (n, N_BODIES, DIM)."""
    positions, velocities = free_dof_to_full_state(theta)
    times = np.linspace(0.0, config.T_MAX, n)
    traj = _integrate_state(positions, velocities, times)
    if traj is None:
        return None
    _, states = traj
    return np.stack([unpack_state(s)[0] for s in states], axis=0)


def _obs_positions(obs_row_block: np.ndarray) -> np.ndarray:
    """Reshape the position block of an observation array to (K, N_BODIES, DIM)."""
    pos = obs_row_block[:, : config.OBS_N_POS]
    return pos.reshape(pos.shape[0], config.N_BODIES, config.DIM)


def run_ppc(
    dataset_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    n_show: int = 6,
    n_draws_plot: int = 30,
    n_systems_cov: int = 150,
    n_draws_cov: int = 60,
    seed: int | None = None,
) -> dict:
    checkpoint_dir = Path(config.CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir)
    results_dir = Path(config.RESULTS_DIR if results_dir is None else results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    raw = np.load(dataset_path, allow_pickle=True)
    # Reproduce train_val_split's exact test indices so the model sees held-out data.
    n = len(data["parameters"])
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * config.TRAIN_VAL_FRACTION))
    test_ids = idx[:n_val]

    test_params = data["parameters"][test_ids]
    test_obs_noisy = raw["observables"][test_ids]

    scales = load_observable_scales(checkpoint_dir / config.OBSERVABLE_SCALES_FILE)
    approx = keras.saving.load_model(str(checkpoint_dir / "three_body_npe.keras"))

    # ---------- qualitative panel: trajectories ----------
    print(f"building PPC trajectory panel for {n_show} systems...")
    show_ids = np.arange(n_show)
    obs_scaled_show = apply_observable_scales_to_dataset(
        {"parameters": test_params[show_ids], "observables": test_obs_noisy[show_ids]}, scales
    )["observables"]
    s = approx.sample(num_samples=n_draws_plot, conditions={"observables": obs_scaled_show})
    key = "parameters" if "parameters" in s else "inference_variables"
    post_show = np.asarray(s[key])  # (n_show, n_draws_plot, n_params)

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for panel, sysid in enumerate(show_ids):
        ax = axes[panel]
        # posterior-predictive trajectories
        for d in range(post_show.shape[1]):
            traj = _dense_trajectory(post_show[sysid, d])
            if traj is None:
                continue
            for b in range(config.N_BODIES):
                ax.plot(traj[:, b, 0], traj[:, b, 1], color=_BODY_COLORS[b],
                        alpha=0.12, lw=0.8)
        # true trajectory (dense)
        true_traj = _dense_trajectory(test_params[sysid])
        if true_traj is not None:
            for b in range(config.N_BODIES):
                ax.plot(true_traj[:, b, 0], true_traj[:, b, 1], color=_BODY_COLORS[b],
                        lw=2.0, label=f"body {b+1}" if panel == 0 else None)
        # observed noisy points
        obs_pos = _obs_positions(test_obs_noisy[sysid])
        for b in range(config.N_BODIES):
            ax.scatter(obs_pos[:, b, 0], obs_pos[:, b, 1], color=_BODY_COLORS[b],
                       edgecolor="black", s=40, zorder=5)
        ax.set_title(f"test system {sysid}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)

    for extra in range(n_show, len(axes)):
        axes[extra].axis("off")
    if n_show > 0:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Posterior predictive check: thin = posterior-sampled trajectories, "
        "thick = truth, dots = observed (x–y projection)", fontsize=12,
    )
    fig.tight_layout()
    fig_path = results_dir / "ppc_trajectories.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")

    # ---------- quantitative: predictive coverage of observed data ----------
    print(f"computing posterior-predictive coverage over {n_systems_cov} systems...")
    n_systems_cov = min(n_systems_cov, len(test_params))
    cov_ids = np.arange(n_systems_cov)
    obs_scaled_cov = apply_observable_scales_to_dataset(
        {"parameters": test_params[cov_ids], "observables": test_obs_noisy[cov_ids]}, scales
    )["observables"]
    s_cov = approx.sample(num_samples=n_draws_cov, conditions={"observables": obs_scaled_cov})
    post_cov = np.asarray(s_cov[key])  # (n_systems_cov, n_draws_cov, n_params)

    inside_count = 0
    total_count = 0
    for i in range(n_systems_cov):
        preds = []
        for d in range(n_draws_cov):
            positions, velocities = free_dof_to_full_state(post_cov[i, d])
            res = simulate_trajectory(positions, velocities)
            if not res.accepted:
                continue
            preds.append(trajectory_to_observation_array(res.positions, res.velocities))
        if len(preds) < 5:
            continue
        preds = np.stack(preds, axis=0)  # (n_ok, K, features)
        lo = np.percentile(preds, 5.0, axis=0)
        hi = np.percentile(preds, 95.0, axis=0)
        observed = test_obs_noisy[i]
        inside = (observed >= lo) & (observed <= hi)
        inside_count += int(inside.sum())
        total_count += int(inside.size)

    ppc_coverage = float(inside_count / total_count) if total_count else float("nan")

    summary = {
        "dataset": str(dataset_path),
        "checkpoint_dir": str(checkpoint_dir),
        "n_systems_coverage": int(n_systems_cov),
        "n_draws_coverage": int(n_draws_cov),
        "ppc_coverage_90": ppc_coverage,
        "note": "fraction of observed features inside the 90% posterior-predictive interval",
    }
    summary_path = results_dir / "ppc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved {summary_path}")
    print(f"posterior-predictive 90% coverage: {ppc_coverage:.3f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Posterior predictive checks for 3-body NPE")
    parser.add_argument("--data", type=str, default="data/train_full_3d.npz")
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--n-show", type=int, default=6)
    parser.add_argument("--n-draws-plot", type=int, default=30)
    parser.add_argument("--n-systems-cov", type=int, default=150)
    parser.add_argument("--n-draws-cov", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_ppc(
        args.data,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        n_show=args.n_show,
        n_draws_plot=args.n_draws_plot,
        n_systems_cov=args.n_systems_cov,
        n_draws_cov=args.n_draws_cov,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
