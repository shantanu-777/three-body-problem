"""
src/generate_data.py — build the offline SBI training dataset.

Why this file exists
--------------------
BayesFlow needs many (parameters, observables) pairs. Each pair is created by:

    theta ~ prior  -->  simulate(theta)  -->  add noise  -->  x

This script automates that loop and saves the result to disk for offline training.

Usage
-----
    ./.venv/bin/python src/generate_data.py --n 500 --output data/train_small.npz
    ./.venv/bin/python src/generate_data.py --n 50000 --output data/train_full.npz --workers 4
    ./.venv/bin/python src/generate_data.py --n 50000 --output data/train_full.npz --resume

Notes
-----
- Large runs are slow because each simulation uses DOP853 with strict tolerances.
- Use --workers to parallelize across CPU cores.
- Checkpoints are saved periodically; use --resume after Ctrl+C to continue.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.priors import (
    add_observation_noise,
    free_dof_to_full_state,
    sample_prior,
    trajectory_to_observation_array,
)
from src.simulator import simulate_trajectory


@dataclass
class GenerationStats:
    n_target: int
    n_accepted: int
    n_proposals: int
    acceptance_rate: float
    elapsed_seconds: float
    output_path: str


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_checkpoint.npz")


def generate_one(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Try to generate one accepted training example.

    Returns (parameters, observables_noisy, observables_clean) or None if rejected.
    """
    theta = sample_prior(rng)
    positions, velocities = free_dof_to_full_state(theta)
    result = simulate_trajectory(positions, velocities)

    if not result.accepted:
        return None

    clean = trajectory_to_observation_array(result.positions, result.velocities)
    noisy_pos, noisy_vel = add_observation_noise(result.positions, result.velocities, rng)
    noisy = trajectory_to_observation_array(noisy_pos, noisy_vel)
    return theta, noisy, clean


def _worker_try_one(seed: int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, int]:
    """Worker entry point: one proposal attempt, returns (sample_or_none, n_proposals=1)."""
    rng = np.random.default_rng(seed)
    return generate_one(rng), 1


def _build_metadata(n_accepted: int, n_proposals: int) -> dict:
    return {
        "n_accepted": n_accepted,
        "n_proposals": n_proposals,
        "acceptance_rate": n_accepted / max(n_proposals, 1),
        "parameter_names": [
            "r1_x", "r1_y", "r2_x", "r2_y", "v1_x", "v1_y", "v2_x", "v2_y",
        ],
        "observable_shape": [config.N_OBS, config.N_BODIES * config.DIM * 2],
        "obs_times": config.OBS_TIMES.tolist(),
        "sigma_x": config.SIGMA_X,
        "sigma_v": config.SIGMA_V,
        "prior_pos_range": config.PRIOR_POS_RANGE,
        "prior_vel_range": config.PRIOR_VEL_RANGE,
    }


def _save_arrays(
    output_path: Path,
    parameters: list[np.ndarray],
    observables: list[np.ndarray],
    observables_clean: list[np.ndarray],
    n_proposals: int,
) -> None:
    param_arr = np.stack(parameters, axis=0)
    obs_arr = np.stack(observables, axis=0)
    clean_arr = np.stack(observables_clean, axis=0)
    metadata = _build_metadata(len(parameters), n_proposals)

    np.savez_compressed(
        output_path,
        parameters=param_arr,
        observables=obs_arr,
        observables_clean=clean_arr,
        metadata=json.dumps(metadata),
    )


def _load_checkpoint(checkpoint_path: Path) -> tuple[list, list, list, int]:
    if not checkpoint_path.exists():
        return [], [], [], 0

    data = np.load(checkpoint_path, allow_pickle=True)
    parameters = list(data["parameters"])
    observables = list(data["observables"])
    observables_clean = list(data["observables_clean"])
    meta = json.loads(str(data["metadata"]))
    n_proposals = int(meta.get("n_proposals", 0))
    print(
        f"resuming from checkpoint: {len(parameters)} accepted, "
        f"{n_proposals} proposals so far"
    )
    return parameters, observables, observables_clean, n_proposals


def generate_dataset(
    n_target: int,
    output_path: str | Path,
    seed: int | None = None,
    max_attempts_factor: float = 200.0,
    workers: int = 1,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> GenerationStats:
    """
    Generate ``n_target`` accepted simulations and save to a compressed .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(output_path)

    base_seed = config.SEED if seed is None else seed
    max_attempts = int(n_target * max_attempts_factor)

    if resume:
        parameters, observables, observables_clean, n_proposals = _load_checkpoint(
            checkpoint_path
        )
    else:
        parameters, observables, observables_clean, n_proposals = [], [], [], 0

    t0 = time.perf_counter()
    seed_counter = base_seed + n_proposals

    def maybe_checkpoint(force: bool = False) -> None:
        n_acc = len(parameters)
        if not force and (n_acc == 0 or n_acc % checkpoint_every != 0):
            return
        _save_arrays(checkpoint_path, parameters, observables, observables_clean, n_proposals)
        if n_acc % checkpoint_every == 0 or force:
            elapsed = time.perf_counter() - t0
            rate = n_acc / max(n_proposals, 1)
            print(
                f"  checkpoint: {n_acc}/{n_target} accepted "
                f"({n_proposals} tries, rate={rate:.3f}, {elapsed:.0f}s elapsed)"
            )

    if workers <= 1:
        rng = np.random.default_rng(base_seed + n_proposals)
        while len(parameters) < n_target and n_proposals < max_attempts:
            n_proposals += 1
            sample = generate_one(rng)
            if sample is None:
                continue
            theta, noisy, clean = sample
            parameters.append(theta)
            observables.append(noisy)
            observables_clean.append(clean)
            maybe_checkpoint()

            if len(parameters) % max(1, n_target // 20) == 0:
                elapsed = time.perf_counter() - t0
                remaining = n_target - len(parameters)
                eta = elapsed / len(parameters) * remaining if parameters else float("inf")
                print(
                    f"  {len(parameters)}/{n_target} accepted "
                    f"({n_proposals} tries, rate={len(parameters)/n_proposals:.3f}, "
                    f"ETA ~{eta/60:.1f} min)"
                )
    else:
        batch_size = workers * 4
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while len(parameters) < n_target and n_proposals < max_attempts:
                needed = n_target - len(parameters)
                n_tasks = min(batch_size, max_attempts - n_proposals, max(needed * 4, workers))
                futures = [
                    pool.submit(_worker_try_one, seed_counter + i)
                    for i in range(n_tasks)
                ]
                seed_counter += n_tasks

                for future in as_completed(futures):
                    sample, n_try = future.result()
                    n_proposals += n_try
                    if sample is None:
                        continue
                    theta, noisy, clean = sample
                    parameters.append(theta)
                    observables.append(noisy)
                    observables_clean.append(clean)

                    if len(parameters) >= n_target:
                        break

                maybe_checkpoint()

                if len(parameters) % max(1, checkpoint_every // 2) == 0 or len(parameters) == n_target:
                    elapsed = time.perf_counter() - t0
                    rate = len(parameters) / max(n_proposals, 1)
                    remaining = n_target - len(parameters)
                    eta = elapsed / len(parameters) * remaining if parameters else float("inf")
                    print(
                        f"  {len(parameters)}/{n_target} accepted "
                        f"({n_proposals} tries, rate={rate:.3f}, ETA ~{eta/60:.1f} min)"
                    )

    elapsed = time.perf_counter() - t0
    n_accepted = len(parameters)

    if n_accepted < n_target:
        maybe_checkpoint(force=True)
        raise RuntimeError(
            f"only generated {n_accepted}/{n_target} accepted samples "
            f"in {n_proposals} attempts; widen the prior, increase max_attempts_factor, "
            f"or resume from {checkpoint_path}"
        )

    _save_arrays(output_path, parameters, observables, observables_clean, n_proposals)
    maybe_checkpoint(force=True)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    stats = GenerationStats(
        n_target=n_target,
        n_accepted=n_accepted,
        n_proposals=n_proposals,
        acceptance_rate=n_accepted / n_proposals,
        elapsed_seconds=elapsed,
        output_path=str(output_path),
    )
    return stats


def estimate_acceptance_rate(n_test: int = 200, seed: int | None = None) -> float:
    """Quick estimate of acceptance rate for the current prior settings."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    accepted = 0
    for _ in range(n_test):
        if generate_one(rng) is not None:
            accepted += 1
    return accepted / n_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-body SBI training data")
    parser.add_argument("--n", type=int, default=500, help="number of accepted simulations")
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_small.npz",
        help="output .npz path",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of parallel worker processes (use 4-8 on a laptop)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="save checkpoint every N accepted samples",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from existing checkpoint file",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="only estimate acceptance rate, do not write a dataset",
    )
    args = parser.parse_args()

    if args.estimate_only:
        rate = estimate_acceptance_rate(n_test=300, seed=args.seed)
        print(f"estimated acceptance rate: {rate:.3f}")
        return

    print(f"Generating {args.n} accepted simulations -> {args.output}")
    if args.workers > 1:
        print(f"using {args.workers} parallel workers")
    stats = generate_dataset(
        args.n,
        args.output,
        seed=args.seed,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    print("done.")
    print(asdict(stats))


if __name__ == "__main__":
    main()
