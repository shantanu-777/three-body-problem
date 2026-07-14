"""
config.py — single source of truth for the 3-body SBI project.

Why this file exists
--------------------
Every module (simulator, prior, data generation, inference, diagnostics) must
agree on the same physical constants and settings. Centralizing them here means
there are no magic numbers scattered across the codebase, and a reader (or a
grader reproducing our results) can see the entire experimental setup at a glance.

Values are grouped into:
  * FINALIZED   — decided and fixed by the project spec or by us.
  * PROVISIONAL — sensible placeholders to be validated/finalized in a later phase.

Units are nondimensional throughout (per the spec): G = 1, total mass = 1,
characteristic length = 1.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants  [FINALIZED — spec]
# ---------------------------------------------------------------------------
G = 1.0                      # gravitational constant (nondimensional)
N_BODIES = 3                 # three-body problem
DIM = 3                      # spatial dimension (2 = planar, 3 = full 3D)

# Three distinct, fixed, LABELED masses in ratio 1:2:3, normalized so sum = 1.
# Body labels must stay consistent between simulator and observation.
MASS_RATIO = np.array([1.0, 2.0, 3.0])
MASSES = MASS_RATIO / MASS_RATIO.sum()      # -> [1/6, 2/6, 3/6] = [0.1667, 0.3333, 0.5]

# Number of independent degrees of freedom we actually infer.
# Nominal DOF = N_BODIES * DIM * 2 (position + velocity).
# COM-at-origin removes DIM, zero-momentum removes DIM.
# We infer bodies 1 & 2 (positions + velocities) and construct body 3 from the
# constraints; see src/priors.py.
#   2D -> (3-1)*2*2 = 8 free DOF
#   3D -> (3-1)*3*2 = 12 free DOF
N_FREE_DOF = (N_BODIES - 1) * DIM * 2

# ---------------------------------------------------------------------------
# Numerical integrator  [FINALIZED — spec]
# ---------------------------------------------------------------------------
SOLVER_METHOD = "DOP853"     # 8th-order explicit Runge-Kutta (scipy.integrate.solve_ivp)
RTOL = 1e-10                 # relative tolerance
ATOL = 1e-12                 # absolute tolerance

# ---------------------------------------------------------------------------
# Force-law softening & rejection  [PROVISIONAL — finalize in Phase 1]
# ---------------------------------------------------------------------------
# Softening epsilon prevents the 1/r^2 force from diverging during close
# approaches (which would stall the integrator). Kept small and FIXED across
# all simulations so it is a consistent part of the generative model.
SOFTENING = 1e-3

# Reject a simulation if any pairwise separation drops below COLLISION_RADIUS
# (near-collision) or any body wanders beyond ESCAPE_RADIUS from the COM (ejection).
COLLISION_RADIUS = 1e-2
ESCAPE_RADIUS = 10.0

# Acceptable relative energy drift |dE/E| over the integration; used as the
# built-in trustworthiness check on the simulator (see Heggie's notes).
MAX_ENERGY_DRIFT = 1e-6

# ---------------------------------------------------------------------------
# Time grid & observations  [set in Phase 2 from Lyapunov analysis]
# ---------------------------------------------------------------------------
# Empirical estimate from src/lyapunov.py on the reference demo configuration:
#   LYAPUNOV_EXPONENT ≈ 0.089  ->  LYAPUNOV_TIME ≈ 11.3
# T_MAX = 10 keeps the full window within ~0.9 Lyapunov times.
# Per Aayush's feedback: observe only 4 points along each trajectory (not 50).
LYAPUNOV_EXPONENT = 0.08866937958425708
LYAPUNOV_TIME = 11.277850422419627

T_MAX = 10.0
N_OBS = 4
OBS_TIMES = np.linspace(0.0, T_MAX, N_OBS)   # e.g. [0, 3.33, 6.67, 10]

# ---------------------------------------------------------------------------
# Observation noise  [set in Phase 3]
# ---------------------------------------------------------------------------
# Independent Gaussian noise added i.i.d. per recorded timestep, with separate
# scales for position and velocity (they carry different units).
# At ~1% of a unit-length scale, noise is visible but does not swamp the signal.
SIGMA_X = 1e-2
SIGMA_V = 1e-2

# Observation layout: N_BODIES*DIM position features + same many velocity
# features per timestep. 2D -> 6+6 = 12; 3D -> 9+9 = 18.
OBS_N_POS = N_BODIES * DIM
OBS_N_VEL = N_BODIES * DIM
OBS_N_FEATURES = OBS_N_POS + OBS_N_VEL

# After block-wise z-scoring, multiply velocity channels by this factor so the
# summary network does not ignore them (they are ~13x noisier relative to scale).
VEL_OBS_EMPHASIS = 8.0

# ---------------------------------------------------------------------------
# Prior over the free DOF (8 in 2D, 12 in 3D)  [set in Phase 3]
# ---------------------------------------------------------------------------
# Uniform prior on bodies 1 & 2 (positions + velocities), with body 3 constructed
# from COM + zero-momentum constraints. Rejection during simulation further
# truncates to bounded, non-colliding orbits (the effective prior for SBI).
#
# Bounds chosen so bodies start separated but not too far (avoid immediate escape).
PRIOR_POS_RANGE = (-2.0, 2.0)
PRIOR_VEL_RANGE = (-0.15, 0.15)

# Cheap prior-level rejection: minimum pairwise distance at t=0.
MIN_INITIAL_SEPARATION = 0.15
MAX_PRIOR_PROPOSALS = 10_000

# Dataset sizes
DATASET_SIZE_DEV = 500
DATASET_SIZE_FULL = 50_000

# ---------------------------------------------------------------------------
# BayesFlow training  [Phase 4/5]
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 32
TRAIN_VAL_FRACTION = 0.1
TRAIN_NUM_DIAG_SAMPLES = 1000
SUMMARY_DIM = 32
SUMMARY_EMBED_DIMS = (128, 128)
SUMMARY_NUM_HEADS = (8, 8)
INFERENCE_FLOW_DEPTH = 8
LEARNING_RATE = 5e-4

# Which inference (generative) network to use for the posterior:
#   "coupling"     -> CouplingFlow (discrete normalizing flow; fast sampling)
#   "flowmatching" -> FlowMatching (continuous flow trained by flow matching;
#                     ODE-based sampling, often more flexible but slower to sample)
# We compare both on the same data; see inference.py / diagnostics.py --inference-network.
INFERENCE_NETWORK = "coupling"
FLOWMATCHING_USE_OPTIMAL_TRANSPORT = True
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
OBSERVABLE_SCALES_FILE = "observable_scales.json"

# ---------------------------------------------------------------------------
# Reproducibility & backend
# ---------------------------------------------------------------------------
SEED = 42
KERAS_BACKEND = "jax"        # set os.environ["KERAS_BACKEND"] before importing bayesflow
