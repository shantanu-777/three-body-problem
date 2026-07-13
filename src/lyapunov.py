"""
src/lyapunov.py — estimate chaos timescale and choose observation times.

Why this file exists
--------------------
The project spec requires us to estimate the **Lyapunov exponent** (the rate at
which two nearby trajectories diverge exponentially) and to place observation
times where the trajectory still carries information about the initial
conditions — roughly within the first few **Lyapunov times** (1 / lambda).

What problem it solves
----------------------
Chaotic systems have a characteristic predictability horizon. If we observe for
too short a time, the data may not distinguish different initial conditions; if
we observe for extremely long times, noise and chaos can dominate. This module
turns a qualitative "the system is chaotic" statement into a quantitative time
grid for the simulator and BayesFlow.

Inputs
------
- A reference initial state (positions, velocities) in the COM frame
- A perturbation size and integration horizon

Outputs
------
- Estimated finite-time Lyapunov exponent lambda
- Lyapunov time tau_L = 1 / lambda
- Recommended observation schedule (t_max, obs_times)

How it connects to the rest of the pipeline
-------------------------------------------
- Uses the Phase 1 simulator's physics (same equations, same integrator).
- Writes recommended values into config.py (T_MAX, OBS_TIMES) after estimation.
- Phase 3 (generate_data.py) reads config.OBS_TIMES when recording trajectories.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.simulator import (
    compute_accelerations,
    make_demo_initial_state,
    pack_state,
    simulate_trajectory,
    unpack_state,
)


@dataclass(frozen=True)
class LyapunovResult:
    """Finite-time Lyapunov estimate from a reference/perturbed pair."""

    lyapunov_exponent: float
    lyapunov_time: float
    perturbation_size: float
    fit_start_time: float
    fit_end_time: float
    fit_r_squared: float
    times: np.ndarray
    separations: np.ndarray
    accepted: bool
    reason: str


def construct_full_state(
    r1: np.ndarray,
    r2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full 3-body state from bodies 1 and 2, enforcing COM + zero momentum.

    Body 3 is not free — it is determined by the constraints:

        r3 = -(m1 r1 + m2 r2) / m3
        v3 = -(m1 v1 + m2 v2) / m3
    """
    m1, m2, m3 = config.MASSES
    r3 = -(m1 * r1 + m2 * r2) / m3
    v3 = -(m1 * v1 + m2 * v2) / m3
    positions = np.stack([r1, r2, r3], axis=0)
    velocities = np.stack([v1, v2, v3], axis=0)
    return positions, velocities


def perturb_initial_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    perturbation_size: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a nearby initial state by perturbing bodies 1 and 2, then reconstruct body 3.

    The perturbation is small (order ``perturbation_size``) so that the linear
    regime of exponential divergence can be observed before saturation.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    direction = rng.normal(size=(2, config.DIM))
    direction /= np.linalg.norm(direction)

    r1 = positions[0] + perturbation_size * direction[0]
    r2 = positions[1] + perturbation_size * direction[1]
    v1 = velocities[0] + perturbation_size * direction[0] * 0.1
    v2 = velocities[1] + perturbation_size * direction[1] * 0.1
    return construct_full_state(r1, r2, v1, v2)


def _integrate_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    t_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Integrate one trajectory on a dense time grid (no collision/escape events).

    Returns (times, states) or None if the integrator fails.
    """
    y0 = pack_state(positions, velocities)

    def derivatives(t: float, state: np.ndarray) -> np.ndarray:
        del t
        pos, vel = unpack_state(state)
        acc = compute_accelerations(pos)
        return pack_state(vel, acc)

    solution = solve_ivp(
        fun=derivatives,
        t_span=(0.0, float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method=config.SOLVER_METHOD,
        rtol=config.RTOL,
        atol=config.ATOL,
    )
    if not solution.success:
        return None
    return solution.t, solution.y.T


def phase_space_separation(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Euclidean distance between two flattened 12-dimensional states."""
    return float(np.linalg.norm(state_a - state_b))


def estimate_lyapunov_exponent(
    positions: np.ndarray,
    velocities: np.ndarray,
    t_max: float | None = None,
    n_times: int = 500,
    perturbation_size: float = 1e-6,
    fit_fraction: tuple[float, float] = (0.05, 0.5),
    seed: int | None = None,
) -> LyapunovResult:
    """
    Estimate a finite-time Lyapunov exponent from two nearby trajectories.

    Method (intuition first)
    ------------------------
    Start two trajectories whose initial conditions differ by a tiny amount d0.
    In a chaotic regime, their separation grows roughly like

        d(t) ~ d0 * exp(lambda * t)

    so log d(t) is approximately a straight line with slope lambda.

    We integrate both trajectories on a dense time grid, compute d(t), and fit
    a line to log d(t) over an early time window (before saturation or collision).
    """
    t_max = config.T_MAX if t_max is None else t_max
    times = np.linspace(0.0, t_max, n_times)

    reference = simulate_trajectory(positions, velocities, t_eval=times, t_max=t_max)
    if not reference.accepted:
        return LyapunovResult(
            lyapunov_exponent=np.nan,
            lyapunov_time=np.nan,
            perturbation_size=perturbation_size,
            fit_start_time=np.nan,
            fit_end_time=np.nan,
            fit_r_squared=np.nan,
            times=times,
            separations=np.full(n_times, np.nan),
            accepted=False,
            reason=f"reference trajectory rejected: {reference.reason}",
        )

    perturbed_pos, perturbed_vel = perturb_initial_state(
        positions, velocities, perturbation_size, seed=seed
    )
    ref_traj = _integrate_state(positions, velocities, times)
    pert_traj = _integrate_state(perturbed_pos, perturbed_vel, times)
    if ref_traj is None or pert_traj is None:
        return LyapunovResult(
            lyapunov_exponent=np.nan,
            lyapunov_time=np.nan,
            perturbation_size=perturbation_size,
            fit_start_time=np.nan,
            fit_end_time=np.nan,
            fit_r_squared=np.nan,
            times=times,
            separations=np.full(n_times, np.nan),
            accepted=False,
            reason="integrator failed during Lyapunov pair integration",
        )

    _, ref_states = ref_traj
    _, pert_states = pert_traj
    separations = np.array(
        [phase_space_separation(a, b) for a, b in zip(ref_states, pert_states)],
        dtype=float,
    )

    d0 = separations[0]
    if d0 <= 0.0:
        return LyapunovResult(
            lyapunov_exponent=np.nan,
            lyapunov_time=np.nan,
            perturbation_size=perturbation_size,
            fit_start_time=np.nan,
            fit_end_time=np.nan,
            fit_r_squared=np.nan,
            times=times,
            separations=separations,
            accepted=False,
            reason="zero initial separation between trajectories",
        )

    # Fit log(d/d0) = lambda * t on an early window (avoid transient + saturation).
    fit_start = fit_fraction[0] * t_max
    fit_end = fit_fraction[1] * t_max
    mask = (times >= fit_start) & (times <= fit_end) & (separations > d0)
    if np.count_nonzero(mask) < 5:
        return LyapunovResult(
            lyapunov_exponent=np.nan,
            lyapunov_time=np.nan,
            perturbation_size=perturbation_size,
            fit_start_time=fit_start,
            fit_end_time=fit_end,
            fit_r_squared=np.nan,
            times=times,
            separations=separations,
            accepted=False,
            reason="not enough points in linear fit window",
        )

    x = times[mask]
    y = np.log(separations[mask] / d0)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if slope <= 0.0:
        return LyapunovResult(
            lyapunov_exponent=slope,
            lyapunov_time=np.inf,
            perturbation_size=perturbation_size,
            fit_start_time=fit_start,
            fit_end_time=fit_end,
            fit_r_squared=r_squared,
            times=times,
            separations=separations,
            accepted=False,
            reason="fitted exponent is non-positive (not in chaotic regime)",
        )

    return LyapunovResult(
        lyapunov_exponent=float(slope),
        lyapunov_time=float(1.0 / slope),
        perturbation_size=perturbation_size,
        fit_start_time=fit_start,
        fit_end_time=fit_end,
        fit_r_squared=float(r_squared),
        times=times,
        separations=separations,
        accepted=True,
        reason="ok",
    )


def recommend_observation_schedule(
    lyapunov_time: float,
    n_obs: int | None = None,
    t_max: float | None = None,
    t_max_in_lyapunov_times: float = 5.0,
    min_obs_in_first_lyapunov_time: int = 8,
) -> tuple[float, np.ndarray]:
    """
    Build an observation grid from the Lyapunov time.

    Strategy
    --------
    - Set T_MAX to ``t_max`` if provided, otherwise ``t_max_in_lyapunov_times * tau_L``.
    - Place at least ``min_obs_in_first_lyapunov_time`` points in [0, tau_L],
      then fill the rest up to T_MAX.

    This satisfies the spec: "several observation times fall within the first
    Lyapunov time or so."
    """
    n_obs = config.N_OBS if n_obs is None else n_obs
    tau = lyapunov_time
    if t_max is None:
        t_max = t_max_in_lyapunov_times * tau

    n_early = min(min_obs_in_first_lyapunov_time, n_obs - 1)
    n_late = n_obs - n_early

    early = np.linspace(0.0, tau, n_early, endpoint=True)
    if n_late > 0:
        late = np.linspace(tau, t_max, n_late + 1, endpoint=True)[1:]
        obs_times = np.concatenate([early, late])
    else:
        obs_times = early

    return t_max, obs_times


if __name__ == "__main__":
    pos, vel = make_demo_initial_state()
    result = estimate_lyapunov_exponent(pos, vel)

    print("accepted:", result.accepted)
    print("reason:", result.reason)
    print("lambda:", result.lyapunov_exponent)
    print("Lyapunov time tau_L:", result.lyapunov_time)
    print("fit R^2:", result.fit_r_squared)

    if result.accepted:
        t_max, obs_times = recommend_observation_schedule(
            result.lyapunov_time,
            t_max=min(10.0, 5.0 * result.lyapunov_time),
        )
        n_in_first = int(np.sum(obs_times <= result.lyapunov_time))
        print("recommended T_MAX:", t_max)
        print("recommended N_OBS:", len(obs_times))
        print("observations in first Lyapunov time:", n_in_first)
        print("first few obs times:", obs_times[:5])
