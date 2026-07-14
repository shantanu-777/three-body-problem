# Inferring Initial Conditions in the 3-Body Problem

Simulation-Based Inference (SBI) with **BayesFlow** for the planar (2D) Newtonian
three-body problem. We infer the **posterior distribution over the initial
conditions** (positions and velocities at *t = 0*) of a chaotic three-body system
from a noisy observation of its trajectory.

TU Dortmund University — Simulation Based Inference, Final Project (Topic 7).

> The full plan and design rationale are in `PROJECT_PLAN.pdf`.

## Research questions

- **Primary:** How accurately can SBI recover the initial conditions of a chaotic
  three-body system, and is the recovered posterior well calibrated?
- **Secondary:** How do posterior uncertainty and shape (width, possible
  multimodality) change with the degree of chaos and the length of the observed
  trajectory relative to the Lyapunov time?

The scientific point (per the spec): for a chaotic system a **wide, possibly
multimodal, but well-calibrated posterior is the correct result, not a failure.**

## Setup

Requires Python 3.13 (a 3.13 interpreter is used to build an isolated venv).

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

BayesFlow 2.x runs on Keras 3, which needs a backend (we use JAX). Set it before
importing bayesflow:

```bash
export KERAS_BACKEND=jax
```

## Project structure

```
config.py            # single source of truth: constants, tolerances, noise, priors
src/
  simulator.py       # Phase 1: initial conditions -> trajectory (+ energy check, rejection)
  lyapunov.py        # Phase 2: estimate Lyapunov exponent, choose observation grid
  priors.py          # Phase 3: sample the 8 free DOF, construct full 3-body state
  generate_data.py   # Phase 3: build & save the offline training dataset
  inference.py       # Phase 4: adapter + summary/inference networks + training
  diagnostics.py     # Phase 5: SBC, posterior contraction, recovery, PPCs
experiments/         # Phase 6: observation-length / chaos sweeps
data/                # saved simulations (gitignored)
notebooks/           # exploration + figures for slides/report
```

## Status

- [x] Phase 0 — environment & scaffold
- [x] Phase 1 — simulator
- [x] Phase 2 — Lyapunov timescale
- [x] Phase 3 — priors + dataset (`data/train_full.npz` target: 50k, **4 obs points/trajectory**)
- [x] Phase 4 — BayesFlow training (`src/inference.py`, checkpoint in `checkpoints/`)
- [x] Phase 5 — diagnostics (`src/diagnostics.py`, figures in `results/`)
- [ ] Phase 6 — experiment
- [ ] Phase 7 — write-up
