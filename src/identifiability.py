"""
src/identifiability.py — test how much each parameter affects the trajectory.

Why this file exists
--------------------
Before blaming the neural network, we ask a physics question:

    If I slightly change one initial condition, how much does the observed
    trajectory change?

Large sensitivity -> parameter should be inferable (if the network is good).
Small sensitivity -> fundamental limit; more data won't fully fix it.

Usage
-----
    ./.venv/bin/python src/identifiability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.priors import (
    PARAMETER_NAMES,
    add_observation_noise,
    free_dof_to_full_state,
    trajectory_to_observation_array,
)
from src.simulator import simulate_trajectory


def _typical_perturbation_sizes() -> np.ndarray:
    """Per-parameter perturbation magnitudes (position vs velocity scales)."""
    sizes = np.empty(config.N_FREE_DOF, dtype=float)
    sizes[0:4] = 0.01 * (config.PRIOR_POS_RANGE[1] - config.PRIOR_POS_RANGE[0])
    sizes[4:8] = 0.01 * (config.PRIOR_VEL_RANGE[1] - config.PRIOR_VEL_RANGE[0])
    return sizes


def trajectory_sensitivity(
    theta: np.ndarray,
    perturbation_sizes: np.ndarray | None = None,
    add_noise: bool = True,
    seed: int | None = None,
) -> dict[str, float]:
    """
    For each parameter, perturb by a small amount and measure ||delta observation||.
    """
    perturbation_sizes = _typical_perturbation_sizes() if perturbation_sizes is None else perturbation_sizes
    rng = np.random.default_rng(config.SEED if seed is None else seed)

    base_pos, base_vel = free_dof_to_full_state(theta)
    base_result = simulate_trajectory(base_pos, base_vel)
    if not base_result.accepted:
        raise RuntimeError("reference simulation rejected; pick another theta")

    base_obs = trajectory_to_observation_array(base_result.positions, base_result.velocities)
    if add_noise:
        noisy_pos, noisy_vel = add_observation_noise(
            base_result.positions, base_result.velocities, rng
        )
        base_obs = trajectory_to_observation_array(noisy_pos, noisy_vel)

    base_vec = base_obs.reshape(-1)
    pos_norm = np.linalg.norm(base_obs[..., : config.OBS_N_POS])
    vel_norm = np.linalg.norm(base_obs[..., config.OBS_N_POS :])

    sensitivities: dict[str, float] = {}
    for i, name in enumerate(PARAMETER_NAMES):
        perturbed = theta.copy()
        perturbed[i] += perturbation_sizes[i]

        pos, vel = free_dof_to_full_state(perturbed)
        result = simulate_trajectory(pos, vel)
        if not result.accepted:
            sensitivities[name] = float("nan")
            continue

        obs = trajectory_to_observation_array(result.positions, result.velocities)
        if add_noise:
            npos, nvel = add_observation_noise(result.positions, result.velocities, rng)
            obs = trajectory_to_observation_array(npos, nvel)

        delta = obs.reshape(-1) - base_vec
        sensitivities[name] = float(np.linalg.norm(delta))

        # Normalized variants for reporting.
        sensitivities[f"{name}_rel_total"] = sensitivities[name] / max(np.linalg.norm(base_vec), 1e-12)
        block = obs[..., : config.OBS_N_POS] if i < 4 else obs[..., config.OBS_N_POS :]
        base_block = base_obs[..., : config.OBS_N_POS] if i < 4 else base_obs[..., config.OBS_N_POS :]
        block_norm = max(np.linalg.norm(base_block), 1e-12)
        sensitivities[f"{name}_rel_block"] = float(
            np.linalg.norm((block - base_block).reshape(-1)) / block_norm
        )

    return sensitivities


def plot_sensitivity(sensitivities: dict[str, float], out_path: Path) -> None:
    """Bar chart of raw observation-space sensitivity."""
    values = [sensitivities[name] for name in PARAMETER_NAMES]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2a6f97" if i < 4 else "#e76f51" for i in range(8)]
    ax.bar(PARAMETER_NAMES, values, color=colors)
    ax.set_ylabel("||Δ observation|| (L2)")
    ax.set_title("Trajectory sensitivity to 1% prior-range parameter perturbations")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_identifiability(
    dataset_path: str | Path = "data/train_small.npz",
    results_dir: str | Path = "results",
    n_cases: int = 20,
    seed: int | None = None,
) -> dict:
    """Average sensitivity over several accepted training examples."""
    data = np.load(dataset_path)
    thetas = data["parameters"]
    rng = np.random.default_rng(config.SEED if seed is None else seed)

    accum = {name: [] for name in PARAMETER_NAMES}
    used = 0
    for _ in range(n_cases * 5):
        if used >= n_cases:
            break
        theta = thetas[rng.integers(0, len(thetas))]
        try:
            sens = trajectory_sensitivity(theta, add_noise=True, seed=int(rng.integers(0, 1_000_000)))
        except RuntimeError:
            continue
        if any(np.isnan(sens[name]) for name in PARAMETER_NAMES):
            continue
        for name in PARAMETER_NAMES:
            accum[name].append(sens[name])
        used += 1

    summary = {
        name: {
            "mean_sensitivity": float(np.mean(accum[name])),
            "std_sensitivity": float(np.std(accum[name])),
        }
        for name in PARAMETER_NAMES
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "identifiability.json").write_text(json.dumps(summary, indent=2))

    mean_values = {name: summary[name]["mean_sensitivity"] for name in PARAMETER_NAMES}
    plot_sensitivity(mean_values, results_dir / "identifiability.png")
    print(f"  saved {results_dir / 'identifiability.png'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter identifiability test")
    parser.add_argument("--data", type=str, default="data/train_small.npz")
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    summary = run_identifiability(args.data, args.results_dir, args.n_cases, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
