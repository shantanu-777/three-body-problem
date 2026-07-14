"""
src/simulator.py — forward physics simulator for the planar 3-body problem.

Why this file exists
--------------------
This is Part 1 of the pipeline: the **generative model's physics engine**.
Given initial positions and velocities, it integrates Newton's equations and
returns the system's trajectory. Every training example in BayesFlow will be
produced by calling this simulator.

What problem it solves
----------------------
We cannot write down a simple formula for p(trajectory | initial conditions)
because the dynamics are chaotic and only available through numerical integration.
The simulator is therefore the definition of the forward map:

    initial conditions  -->  trajectory

Inputs
------
- positions: array of shape (3, 2) — (x, y) for each labeled body
- velocities: array of shape (3, 2) — (vx, vy) for each labeled body

The state must satisfy (within tolerance):
  * center of mass at the origin
  * total momentum zero

Outputs
-------
A SimulationResult containing:
- times, positions, velocities recorded at the observation grid
- whether the run was accepted or rejected (collision / escape / energy drift)
- energy-conservation diagnostics

How it connects to the rest of the pipeline
-------------------------------------------
- Phase 2 (lyapunov.py) calls this to estimate the Lyapunov exponent.
- Phase 3 (priors.py, generate_data.py) samples initial conditions and calls
  this thousands of times to build the offline training dataset.
- Phase 4+ never call this at training time (offline data), but posterior
  predictive checks may call it again to simulate from inferred parameters.
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


@dataclass(frozen=True)
class SimulationResult:
    """Container for one simulation run."""

    accepted: bool
    reason: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    initial_energy: float
    final_energy: float
    energy_drift: float


def pack_state(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Flatten (N_BODIES, DIM) positions and velocities into a flat state vector."""
    return np.concatenate([positions.ravel(), velocities.ravel()])


def unpack_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover (N_BODIES, DIM) positions and velocities from the flat state vector."""
    n_pos = config.N_BODIES * config.DIM
    positions = state[:n_pos].reshape(config.N_BODIES, config.DIM)
    velocities = state[n_pos:].reshape(config.N_BODIES, config.DIM)
    return positions, velocities


def center_of_mass(positions: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Weighted center of mass (a 2D vector)."""
    masses = config.MASSES if masses is None else masses
    return np.average(positions, axis=0, weights=masses)


def total_momentum(velocities: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Total momentum vector (should be zero in the COM frame)."""
    masses = config.MASSES if masses is None else masses
    return np.sum(velocities * masses[:, None], axis=0)


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    g: float | None = None,
    epsilon: float | None = None,
) -> np.ndarray:
    """
    Newtonian gravity with Plummer-style softening.

    For each body i, the acceleration is

        a_i = G * sum_{j != i}  m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

    Softening epsilon prevents the force from diverging when two bodies pass very
    close together, which would otherwise force the integrator to take tiny steps.
    """
    masses = config.MASSES if masses is None else masses
    g = config.G if g is None else g
    epsilon = config.SOFTENING if epsilon is None else epsilon

    n_bodies = positions.shape[0]
    accelerations = np.zeros_like(positions)

    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            displacement = positions[j] - positions[i]
            dist_sq = np.dot(displacement, displacement) + epsilon**2
            accelerations[i] += g * masses[j] * displacement / (dist_sq ** 1.5)

    return accelerations


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray | None = None) -> float:
    """Kinetic energy T = 1/2 * sum_i m_i |v_i|^2."""
    masses = config.MASSES if masses is None else masses
    speeds_sq = np.sum(velocities**2, axis=1)
    return 0.5 * float(np.sum(masses * speeds_sq))


def potential_energy(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    g: float | None = None,
    epsilon: float | None = None,
) -> float:
    """
    Softened gravitational potential energy.

        U = -G * sum_{i<j} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)

    This potential is consistent with the softened force law above.
    """
    masses = config.MASSES if masses is None else masses
    g = config.G if g is None else g
    epsilon = config.SOFTENING if epsilon is None else epsilon

    energy = 0.0
    n_bodies = positions.shape[0]
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dist_sq = np.sum((positions[i] - positions[j]) ** 2) + epsilon**2
            energy -= g * masses[i] * masses[j] / np.sqrt(dist_sq)
    return float(energy)


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray | None = None,
) -> float:
    """Total mechanical energy E = T + U."""
    return kinetic_energy(velocities, masses) + potential_energy(positions, masses)


def min_pairwise_distance(positions: np.ndarray) -> float:
    """Smallest distance between any two distinct bodies."""
    n_bodies = positions.shape[0]
    min_dist = np.inf
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, dist)
    return float(min_dist)


def max_distance_from_com(positions: np.ndarray, masses: np.ndarray | None = None) -> float:
    """Largest distance of any body from the center of mass."""
    masses = config.MASSES if masses is None else masses
    com = center_of_mass(positions, masses)
    return float(np.max(np.linalg.norm(positions - com, axis=1)))


def validate_initial_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    atol: float = 1e-10,
) -> tuple[bool, str]:
    """
    Check COM-at-origin and zero-momentum constraints on the initial state.
    """
    com = center_of_mass(positions)
    momentum = total_momentum(velocities)

    if not np.allclose(com, 0.0, atol=atol):
        return False, f"center of mass not at origin: {com}"

    if not np.allclose(momentum, 0.0, atol=atol):
        return False, f"total momentum not zero: {momentum}"

    return True, "ok"


def _equations_of_motion(
    t: float,
    state: np.ndarray,
    masses: np.ndarray,
    g: float,
    epsilon: float,
) -> np.ndarray:
    """
    First-order ODE system for solve_ivp.

    State layout: [r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y]
    Derivative layout: [v1x, v1y, ..., a1x, a1y, ...]
    """
    del t  # autonomous system — no explicit time dependence
    positions, velocities = unpack_state(state)
    accelerations = compute_accelerations(positions, masses, g, epsilon)
    return pack_state(velocities, accelerations)


def _make_event_functions():
    """Terminal events: stop integration on near-collision or escape."""

    def collision_event(t: float, state: np.ndarray) -> float:
        del t
        positions, _ = unpack_state(state)
        return min_pairwise_distance(positions) - config.COLLISION_RADIUS

    def escape_event(t: float, state: np.ndarray) -> float:
        del t
        positions, _ = unpack_state(state)
        return config.ESCAPE_RADIUS - max_distance_from_com(positions)

    collision_event.terminal = True
    collision_event.direction = -1

    escape_event.terminal = True
    escape_event.direction = -1

    return collision_event, escape_event


def simulate_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    t_eval: np.ndarray | None = None,
    t_max: float | None = None,
    masses: np.ndarray | None = None,
    g: float | None = None,
    epsilon: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    check_initial_constraints: bool = True,
) -> SimulationResult:
    """
    Integrate the 3-body system and record its trajectory.

    Parameters
    ----------
    positions, velocities
        Initial state arrays of shape (3, 2).
    t_eval
        Times at which to record the trajectory. Defaults to config.OBS_TIMES.
    t_max
        Final integration time. Defaults to the last entry in t_eval.

    Returns
    -------
    SimulationResult
        Includes acceptance flag, recorded trajectory, and energy diagnostics.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = config.MASSES if masses is None else masses
    g = config.G if g is None else g
    epsilon = config.SOFTENING if epsilon is None else epsilon
    rtol = config.RTOL if rtol is None else rtol
    atol = config.ATOL if atol is None else atol

    if t_eval is None:
        t_eval = config.OBS_TIMES
    t_eval = np.asarray(t_eval, dtype=float)
    if t_max is None:
        t_max = float(t_eval[-1])

    if check_initial_constraints:
        ok, reason = validate_initial_state(positions, velocities)
        if not ok:
            return SimulationResult(
                accepted=False,
                reason=reason,
                times=t_eval,
                positions=np.full((len(t_eval), config.N_BODIES, config.DIM), np.nan),
                velocities=np.full((len(t_eval), config.N_BODIES, config.DIM), np.nan),
                initial_energy=np.nan,
                final_energy=np.nan,
                energy_drift=np.nan,
            )

    if min_pairwise_distance(positions) < config.COLLISION_RADIUS:
        return _rejected_result(
            "initial configuration too close (collision)",
            positions,
            velocities,
            t_eval,
        )

    if max_distance_from_com(positions, masses) > config.ESCAPE_RADIUS:
        return _rejected_result(
            "initial configuration already escaped",
            positions,
            velocities,
            t_eval,
        )

    y0 = pack_state(positions, velocities)
    initial_energy = total_energy(positions, velocities, masses)

    collision_event, escape_event = _make_event_functions()

    solution = solve_ivp(
        fun=lambda t, y: _equations_of_motion(t, y, masses, g, epsilon),
        t_span=(0.0, t_max),
        y0=y0,
        method=config.SOLVER_METHOD,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=[collision_event, escape_event],
    )

    if not solution.success:
        return _rejected_result(
            f"integrator failed: {solution.message}",
            positions,
            velocities,
            t_eval,
        )

    if solution.t_events[0].size > 0:
        return _rejected_result("collision during integration", positions, velocities, t_eval)

    if solution.t_events[1].size > 0:
        return _rejected_result("body escaped during integration", positions, velocities, t_eval)

    recorded_positions = np.stack(
        [unpack_state(state)[0] for state in solution.y.T],
        axis=0,
    )
    recorded_velocities = np.stack(
        [unpack_state(state)[1] for state in solution.y.T],
        axis=0,
    )

    final_energy = total_energy(recorded_positions[-1], recorded_velocities[-1], masses)
    if abs(initial_energy) < 1e-14:
        energy_drift = abs(final_energy - initial_energy)
    else:
        energy_drift = abs((final_energy - initial_energy) / initial_energy)

    if energy_drift > config.MAX_ENERGY_DRIFT:
        return SimulationResult(
            accepted=False,
            reason=f"energy drift too large: {energy_drift:.2e}",
            times=solution.t,
            positions=recorded_positions,
            velocities=recorded_velocities,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_drift=energy_drift,
        )

    return SimulationResult(
        accepted=True,
        reason="ok",
        times=solution.t,
        positions=recorded_positions,
        velocities=recorded_velocities,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_drift=energy_drift,
    )


def _rejected_result(
    reason: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    t_eval: np.ndarray,
) -> SimulationResult:
    """Build a rejected SimulationResult with NaN trajectory arrays."""
    initial_energy = total_energy(positions, velocities)
    return SimulationResult(
        accepted=False,
        reason=reason,
        times=t_eval,
        positions=np.full((len(t_eval), config.N_BODIES, config.DIM), np.nan),
        velocities=np.full((len(t_eval), config.N_BODIES, config.DIM), np.nan),
        initial_energy=initial_energy,
        final_energy=np.nan,
        energy_drift=np.nan,
    )


def make_demo_initial_state(seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a bounded demo configuration in the COM frame.

    Uses a hand-crafted, well-separated setup that typically survives the
    default integration window without collision. The ``seed`` argument is
    kept for API compatibility but does not change this demo state.
    """
    del seed

    m1, m2, m3 = config.MASSES
    separation = 5.0
    d = config.DIM

    def _vec(values: list[float]) -> np.ndarray:
        """Build a DIM-length vector, padding/truncating the given components."""
        out = np.zeros(d, dtype=float)
        out[: min(d, len(values))] = values[: min(d, len(values))]
        return out

    # Two bodies on the x-axis; in 3D give them a small out-of-plane (z) offset
    # so the demo configuration is genuinely three-dimensional but still bounded.
    r1 = _vec([-separation, 0.0, 1.0])
    r2 = _vec([separation, 0.0, -1.0])
    r3 = -(m1 * r1 + m2 * r2) / m3

    v1 = _vec([0.0, 0.05, 0.01])
    v2 = _vec([0.0, -0.025, -0.005])
    v3 = -(m1 * v1 + m2 * v2) / m3

    positions = np.stack([r1, r2, r3], axis=0)
    velocities = np.stack([v1, v2, v3], axis=0)
    return positions, velocities


if __name__ == "__main__":
    pos, vel = make_demo_initial_state()
    result = simulate_trajectory(pos, vel)

    print("accepted:", result.accepted)
    print("reason:", result.reason)
    print("energy drift:", result.energy_drift)
    print("trajectory shape:", result.positions.shape, result.velocities.shape)
