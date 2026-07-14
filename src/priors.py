"""
src/priors.py — prior over initial conditions and observation noise model.

Why this file exists
--------------------
BayesFlow learns p(parameters | data). The **prior** p(parameters) describes
which initial conditions we consider plausible *before* seeing any trajectory.
This file defines that distribution and the map between our 8 inferred
parameters and the full 3-body state.

What problem it solves
----------------------
We do not infer all 12 position/velocity components directly. Four are fixed by
physics constraints (COM at origin, zero total momentum). We infer 8 numbers
(bodies 1 and 2), then **construct** body 3:

    r3 = -(m1 r1 + m2 r2) / m3
    v3 = -(m1 v1 + m2 v2) / m3

The observation model adds Gaussian noise to the simulated trajectory.

Inputs / outputs
----------------
- Input: 8-vector theta OR full (positions, velocities)
- Output: full initial state, noisy trajectory arrays, log-prior (for uniform prior)

Connects to
-----------
- simulator.py: full state -> trajectory
- generate_data.py: sample theta, simulate, add noise, save training pairs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.simulator import min_pairwise_distance, max_distance_from_com, validate_initial_state


# Axis labels used to build parameter names for any spatial dimension.
_AXIS_LABELS = ("x", "y", "z", "w")


def _build_parameter_names() -> list[str]:
    """
    Names for the inferred free parameters, ordered [r1, r2, v1, v2] with each
    block spanning all spatial axes. In 2D this is 8 names; in 3D, 12.
    """
    axes = _AXIS_LABELS[: config.DIM]
    names: list[str] = []
    for kind in ("r", "v"):
        for body in (1, 2):
            for axis in axes:
                names.append(f"{kind}{body}_{axis}")
    return names


# Names for the inferred parameters (for plots and the report).
PARAMETER_NAMES = _build_parameter_names()


def free_dof_to_full_state(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map the free parameters to full positions and velocities (N_BODIES, DIM) each.

    theta layout: [r1(DIM), r2(DIM), v1(DIM), v2(DIM)]; body 3 is constructed
    from the COM-at-origin and zero-momentum constraints.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (config.N_FREE_DOF,):
        raise ValueError(f"expected theta shape ({config.N_FREE_DOF},), got {theta.shape}")

    d = config.DIM
    r1 = theta[0:d]
    r2 = theta[d : 2 * d]
    v1 = theta[2 * d : 3 * d]
    v2 = theta[3 * d : 4 * d]

    m1, m2, m3 = config.MASSES
    r3 = -(m1 * r1 + m2 * r2) / m3
    v3 = -(m1 * v1 + m2 * v2) / m3

    positions = np.stack([r1, r2, r3], axis=0)
    velocities = np.stack([v1, v2, v3], axis=0)
    return positions, velocities


def full_state_to_free_dof(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Extract the 8 free parameters from a full COM-frame state."""
    return np.concatenate([
        positions[0], positions[1],
        velocities[0], velocities[1],
    ])


def log_prior(theta: np.ndarray) -> float:
    """
    Log-density of the (improper) uniform prior over the box defined in config.

    For a proper truncated prior used in rejection sampling, the normalization
    constant is unknown; this returns 0 inside the box and -inf outside.
    """
    theta = np.asarray(theta, dtype=float)
    lo_p, hi_p = config.PRIOR_POS_RANGE
    lo_v, hi_v = config.PRIOR_VEL_RANGE

    n_pos = 2 * config.DIM
    pos = theta[:n_pos]
    vel = theta[n_pos:]

    if np.any(pos < lo_p) or np.any(pos > hi_p):
        return -np.inf
    if np.any(vel < lo_v) or np.any(vel > hi_v):
        return -np.inf
    return 0.0


def is_valid_prior_draw(positions: np.ndarray, velocities: np.ndarray) -> bool:
    """
    Cheap checks before running the expensive integrator.

    Rejects proposals that violate constraints or are obviously unbounded.
    """
    ok, _ = validate_initial_state(positions, velocities)
    if not ok:
        return False
    if min_pairwise_distance(positions) < config.MIN_INITIAL_SEPARATION:
        return False
    if max_distance_from_com(positions) > config.ESCAPE_RADIUS:
        return False
    return True


def sample_prior(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Draw one sample from the uniform prior over the 8 free DOF.

    Uses rejection at the prior level (minimum separation, COM constraints)
    so that expensive simulations are not wasted on obviously bad states.
    """
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    lo_p, hi_p = config.PRIOR_POS_RANGE
    lo_v, hi_v = config.PRIOR_VEL_RANGE

    n_pos = 2 * config.DIM
    for _ in range(config.MAX_PRIOR_PROPOSALS):
        theta = np.empty(config.N_FREE_DOF, dtype=float)
        theta[:n_pos] = rng.uniform(lo_p, hi_p, size=n_pos)
        theta[n_pos:] = rng.uniform(lo_v, hi_v, size=n_pos)
        positions, velocities = free_dof_to_full_state(theta)
        if is_valid_prior_draw(positions, velocities):
            return theta

    raise RuntimeError(
        f"could not draw a valid prior sample in {config.MAX_PRIOR_PROPOSALS} tries; "
        "consider widening PRIOR_*_RANGE or lowering MIN_INITIAL_SEPARATION"
    )


def trajectory_to_observation_array(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """
    Flatten a simulated trajectory into a (K, 12) array for BayesFlow.

    Each row is one observation time, features ordered as
    [r1_x, r1_y, r2_x, r2_y, r3_x, r3_y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y].
    """
    pos_flat = positions.reshape(positions.shape[0], -1)
    vel_flat = velocities.reshape(velocities.shape[0], -1)
    return np.concatenate([pos_flat, vel_flat], axis=1)


def add_observation_noise(
    positions: np.ndarray,
    velocities: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add independent Gaussian noise i.i.d. per timestep (spec observation model).

        x_hat = x + sigma_x * eps_x
        v_hat = v + sigma_v * eps_v
    """
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    noisy_pos = positions + rng.normal(0.0, config.SIGMA_X, size=positions.shape)
    noisy_vel = velocities + rng.normal(0.0, config.SIGMA_V, size=velocities.shape)
    return noisy_pos, noisy_vel
