"""
src/build_slides.py — build the space-themed presentation as a single PDF.

Why this file exists
--------------------
The project brief requires a PDF slide deck (10 minutes, group of 1-2) that:
  * first slide: title + group members' names,
  * introduces the topic,
  * explains the statistical model and how it is fit with BayesFlow,
  * discusses results (with captioned, labelled figures),
  * last slide: TL;DR / take-home message + contact info,
  * no "thank you" slide.

We render every slide with matplotlib and stitch them into one PDF, so the deck
is fully self-contained and reproducible (no LaTeX / PowerPoint needed). The
theme is "deep space" to match the celestial-mechanics topic.

Edit the GROUP / MEMBERS / CONTACT constants below before submitting.

Usage
-----
    ./.venv/bin/python src/build_slides.py
    -> results/slides.pdf  (+ per-slide PNG previews in results/slides_preview/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# EDIT THESE BEFORE SUBMITTING
# ---------------------------------------------------------------------------
GROUP_NUMBER = "XX"
MEMBERS = ["Your Name", "Teammate Name"]
CONTACT = "firstname.lastname@tu-dortmund.de"

RESULTS = _ROOT / "results"
OUT_PDF = RESULTS / "slides.pdf"
PREVIEW_DIR = RESULTS / "slides_preview"

# ---------------------------------------------------------------------------
# Space theme palette
# ---------------------------------------------------------------------------
BG = "#05070f"          # near-black deep space
PANEL = "#0d1424"       # slightly lighter panel
INK = "#eaf0ff"         # near-white text
MUTED = "#9fb0d0"       # muted blue-grey
ACCENT = "#ffd166"      # warm gold (stars/sun)
ACCENT2 = "#5ad2ff"     # cyan (accent)
GOOD = "#8ef0a6"        # green

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": MUTED,
})

W, H = 13.333, 7.5      # 16:9 inches

_STAR_RNG = np.random.default_rng(7)
_STARS = {
    "x": _STAR_RNG.uniform(0, 1, 340),
    "y": _STAR_RNG.uniform(0, 1, 340),
    "s": _STAR_RNG.uniform(0.5, 9.0, 340),
    "a": _STAR_RNG.uniform(0.25, 0.95, 340),
}


def new_slide():
    """Create a figure with the starry space background."""
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor(BG)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_facecolor(BG)
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    # subtle vertical glow gradient near the top
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    bg.imshow(grad, extent=[0, 1, 0, 1], aspect="auto", cmap="Blues_r",
              alpha=0.06, zorder=0)
    bg.scatter(_STARS["x"], _STARS["y"], s=_STARS["s"], c="white",
               alpha=_STARS["a"], zorder=1, edgecolors="none")
    # a few gold "stars"
    bg.scatter(_STARS["x"][:40], _STARS["y"][:40], s=_STARS["s"][:40] * 1.4,
               c=ACCENT, alpha=0.5, zorder=1, edgecolors="none")
    return fig, bg


def footer(bg, idx, total):
    bg.plot([0.06, 0.94], [0.055, 0.055], color=MUTED, lw=0.8, alpha=0.4)
    bg.text(0.06, 0.028, "Topic 7 · Inferring Initial Conditions in the 3-Body Problem",
            color=MUTED, fontsize=8, va="center")
    bg.text(0.94, 0.028, f"{idx}/{total}", color=MUTED, fontsize=8,
            va="center", ha="right")


def title_line(bg, text, y=0.88, size=26, color=ACCENT):
    bg.text(0.06, y, text, color=color, fontsize=size, fontweight="bold", va="center")
    bg.plot([0.062, 0.30], [y - 0.055, y - 0.055], color=ACCENT2, lw=2.5, alpha=0.9)


def bullets(bg, items, x=0.07, y=0.72, dy=0.083, size=15, color=INK, bullet=ACCENT2):
    for i, it in enumerate(items):
        yy = y - i * dy
        indent = 0.0
        txt = it
        if isinstance(it, tuple):
            indent, txt = it
        bg.text(x + indent, yy, "•", color=bullet, fontsize=size, va="center", ha="left")
        bg.text(x + indent + 0.022, yy, txt, color=color, fontsize=size, va="center", ha="left")


def panel_image(fig, bg, path: Path, rect, caption=None, title=None):
    """Place an image on a rounded panel; rect = [l, b, w, h] in fig coords."""
    l, b, w, h = rect
    # panel background
    pad = 0.012
    fancy = FancyBboxPatch(
        (l - pad, b - pad), w + 2 * pad, h + 2 * pad,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=1.0, edgecolor=ACCENT2, facecolor=PANEL, alpha=0.95,
        transform=fig.transFigure, zorder=2,
    )
    bg.add_patch(fancy)
    if title:
        bg.text(l, b + h + pad + 0.015, title, color=ACCENT2, fontsize=12,
                fontweight="bold", va="bottom", ha="left")
    if path.exists():
        ax = fig.add_axes([l, b, w, h], zorder=3)
        ax.imshow(mpimg.imread(str(path)))
        ax.axis("off")
    else:
        bg.text(l + w / 2, b + h / 2, f"[missing {path.name}]",
                color=MUTED, fontsize=11, ha="center", va="center")
    if caption:
        bg.text(l + w / 2, b - pad - 0.028, caption, color=MUTED, fontsize=9.5,
                ha="center", va="top", style="italic")


def hero_trajectory(bg, rect):
    """Draw a decorative 3-body orbit directly on the slide (space theme)."""
    import config
    from src.lyapunov import _integrate_state
    from src.simulator import make_demo_initial_state, unpack_state

    l, b, w, h = rect
    ax = plt.gcf().add_axes([l, b, w, h], zorder=3)
    ax.set_facecolor("none")
    pos, vel = make_demo_initial_state()
    times = np.linspace(0.0, 18.0, 500)
    traj = _integrate_state(pos, vel, times)
    colors = [ACCENT, GOOD, ACCENT2]
    if traj is not None:
        _, states = traj
        P = np.stack([unpack_state(s)[0] for s in states], axis=0)
        for bd in range(config.N_BODIES):
            ax.plot(P[:, bd, 0], P[:, bd, 1], color=colors[bd], lw=1.6, alpha=0.9)
            ax.scatter(P[-1, bd, 0], P[-1, bd, 1], s=90, color=colors[bd],
                       edgecolors="white", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def build():
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    slides = []

    # ---- 1. Title ----
    def s_title(fig, bg):
        bg.text(0.06, 0.80, "Inferring Initial Conditions", color=INK, fontsize=40,
                fontweight="bold", va="center")
        bg.text(0.06, 0.70, "in the 3-Body Problem", color=ACCENT, fontsize=40,
                fontweight="bold", va="center")
        bg.text(0.06, 0.60, "Amortized Bayesian inference with BayesFlow",
                color=ACCENT2, fontsize=18, va="center")
        bg.text(0.06, 0.30, "  ·  ".join(MEMBERS), color=INK, fontsize=17, va="center")
        bg.text(0.06, 0.24, f"Group {GROUP_NUMBER} · Simulation-Based Inference · TU Dortmund",
                color=MUTED, fontsize=13, va="center")
        hero_trajectory(bg, [0.60, 0.22, 0.34, 0.52])
        bg.text(0.77, 0.18, "a chaotic 3-body orbit", color=MUTED, fontsize=9,
                style="italic", ha="center")

    # ---- 2. Introduction ----
    def s_intro(fig, bg):
        title_line(bg, "The 3-body problem & chaos")
        bullets(bg, [
            "Three masses attract each other under Newtonian gravity.",
            "No closed-form solution; the system is generically chaotic.",
            "Nearby initial conditions diverge exponentially (Lyapunov time).",
            "Forward map (initial state → trajectory) is extremely sensitive.",
            "Question: can we run it backwards — recover the start from a",
            (0.022, "noisy, partial observation of the motion?"),
        ], y=0.72)
        hero_trajectory(bg, [0.62, 0.14, 0.33, 0.52])
        bg.text(0.785, 0.12, "bodies 1:2:3, COM frame", color=MUTED, fontsize=9,
                style="italic", ha="center")

    # ---- 3. The task / SBI framing ----
    def s_task(fig, bg):
        title_line(bg, "The inference task")
        bullets(bg, [
            "Goal: posterior p(initial conditions | noisy trajectory).",
            "Amortized Neural Posterior Estimation (NPE) with BayesFlow.",
            "Infer positions + velocities of the three bodies at t = 0.",
            "",
            "Central scientific point:",
            (0.022, "for a chaotic system a WIDE but WELL-CALIBRATED posterior"),
            (0.022, "is the correct result — not a failure of the method."),
        ], y=0.72)
        bg.text(0.07, 0.12, "We test exactly this: does the posterior widen with chaos"
                            " and stay calibrated?", color=ACCENT2, fontsize=13, va="center")

    # ---- 4. Statistical model / simulator ----
    def s_model(fig, bg):
        title_line(bg, "Statistical model — the simulator")
        bullets(bg, [
            "Nondimensional units: G = 1, total mass = 1, length = 1.",
            "Distinct labelled masses in ratio 1 : 2 : 3.",
            "COM frame: centre of mass at origin, total momentum = 0.",
            "Integrator: solve_ivp DOP853, rtol=1e-10, atol=1e-12.",
            "Softening ε to tame close approaches; reject collision / escape.",
            "Energy conservation (|ΔE/E| < 1e-6) as a built-in sanity check.",
        ], y=0.72, dy=0.088)
        bullets(bg, [
            "Prior: positions ~ U[-2, 2], velocities ~ U[-0.15, 0.15]",
            "Noise: Gaussian, i.i.d. per observed time (σ = 0.01).",
            "Infer bodies 1 & 2; body 3 fixed by COM + momentum.",
        ], x=0.55, y=0.60, dy=0.09, size=13, color=MUTED, bullet=ACCENT)

    # ---- 5. Observation model + Lyapunov ----
    def s_obs(fig, bg):
        title_line(bg, "Observation model & timescale")
        bullets(bg, [
            "Observe the trajectory, not just the endpoint.",
            "K = 4 snapshots at t = [0, 3.3, 6.7, 10].",
            "Lyapunov exponent λ ≈ 0.089  →  Lyapunov time τ ≈ 11.3.",
            "All observation times fall within the first Lyapunov time,",
            (0.022, "where the information about the initial state actually is."),
            "8 free parameters in 2D → 12 in 3D.",
        ], y=0.72)

    # ---- 6. Approximator / training ----
    def s_net(fig, bg):
        title_line(bg, "Approximator & training (BayesFlow)")
        bullets(bg, [
            "Summary network: TimeSeriesTransformer",
            (0.022, "summary_dim=32, embed (128,128), heads (8,8)."),
            "Inference network: CouplingFlow (depth 8)  —  and FlowMatching.",
            "Separate position / velocity standardization (velocity ×8 emphasis).",
            "Offline training: 50,000 simulations, ~83% accepted (3D).",
            "100 epochs, batch 32, Adam, lr 5e-4, 45k train / 5k val.",
        ], y=0.72)
        panel_image(fig, bg, RESULTS / "losses.png", [0.60, 0.16, 0.34, 0.42],
                    caption="Training & validation loss", title="Convergence")

    # ---- 7. Recovery ----
    def s_recovery(fig, bg):
        title_line(bg, "Result — parameter recovery (3D)")
        panel_image(fig, bg, RESULTS / "recovery.png", [0.07, 0.20, 0.62, 0.60],
                    caption="Posterior-mean estimate vs ground truth for all 12 parameters.")
        bullets(bg, [
            "Positions: r = 1.00.",
            "Velocities: r ≈ 0.99.",
            "z-axis recovers as",
            (0.022, "well as x and y."),
            "50k samples fixed the",
            (0.022, "earlier weak velocity."),
        ], x=0.72, y=0.66, dy=0.075, size=13)

    # ---- 8. Calibration / SBC ----
    def s_calib(fig, bg):
        title_line(bg, "Result — calibration (SBC)")
        panel_image(fig, bg, RESULTS / "calibration_ecdf.png", [0.07, 0.20, 0.62, 0.60],
                    caption="Simulation-based calibration: rank ECDFs inside the confidence band.")
        bullets(bg, [
            "Ranks are uniform →",
            (0.022, "posterior is calibrated."),
            "90% credible-interval",
            (0.022, "coverage ≈ 0.95."),
            "Honest uncertainty,",
            (0.022, "not overconfident."),
        ], x=0.72, y=0.66, dy=0.075, size=13)

    # ---- 9. PPC ----
    def s_ppc(fig, bg):
        title_line(bg, "Result — posterior predictive check")
        panel_image(fig, bg, RESULTS / "ppc_trajectories.png", [0.07, 0.22, 0.62, 0.56],
                    caption="Thin = posterior-sampled trajectories, thick = truth, dots = observed.")
        bullets(bg, [
            "Sample ICs from the",
            (0.022, "posterior, simulate"),
            (0.022, "forward."),
            "Predicted trajectories",
            (0.022, "hug the truth &"),
            (0.022, "hit the data."),
            "Predictive 90%",
            (0.022, "coverage = 0.89."),
        ], x=0.72, y=0.68, dy=0.070, size=13)

    # ---- 10. Network comparison ----
    def s_compare(fig, bg):
        title_line(bg, "CouplingFlow vs FlowMatching")
        panel_image(fig, bg, RESULTS / "compare_networks.png", [0.07, 0.30, 0.86, 0.46],
                    caption="Per-parameter recovery (left) and 90% coverage (right) for both networks.")
        bullets(bg, [
            "Recovery: identical (r ≈ 0.998 vs 0.997).   "
            "Calibration: FlowMatching slightly better.   "
            "Speed: CouplingFlow ~100× faster to sample.",
        ], x=0.07, y=0.20, dy=0.07, size=13, color=INK, bullet=ACCENT)

    # ---- 11. Chaos & uncertainty ----
    def s_chaos(fig, bg):
        title_line(bg, "Uncertainty reflects the physics")
        panel_image(fig, bg, RESULTS / "chaos_analysis.png", [0.05, 0.30, 0.44, 0.46],
                    caption="More chaotic → wider velocity posterior; calibration holds.",
                    title="Posterior width vs chaos")
        panel_image(fig, bg, RESULTS / "experiment_noise.png", [0.53, 0.30, 0.44, 0.46],
                    caption="Noisier data → wider posterior; overconfident only at extreme σ.",
                    title="Posterior width vs noise")
        bullets(bg, [
            "The posterior gets wider for more chaotic systems and for noisier"
            " data — exactly the expected, honest behaviour.",
        ], x=0.06, y=0.20, dy=0.06, size=13, color=INK, bullet=ACCENT)

    # ---- 12. TL;DR ----
    def s_tldr(fig, bg):
        title_line(bg, "Take-home message", color=ACCENT)
        bullets(bg, [
            "BayesFlow recovers 3-body initial conditions with near-perfect",
            (0.022, "accuracy and well-calibrated posteriors — in 2D and 3D."),
            "Uncertainty honestly grows with chaos and with observation noise.",
            "CouplingFlow = best speed/accuracy; FlowMatching = competitive,",
            (0.022, "slightly better calibrated."),
            "A wide, calibrated posterior is the correct answer for a chaotic system.",
        ], y=0.70, dy=0.088, size=16)
        fancy = FancyBboxPatch((0.06, 0.13), 0.88, 0.10,
                               boxstyle="round,pad=0.01,rounding_size=0.02",
                               linewidth=1.0, edgecolor=ACCENT, facecolor=PANEL,
                               transform=fig.transFigure, alpha=0.9)
        bg.add_patch(fancy)
        bg.text(0.5, 0.18, f"Group {GROUP_NUMBER}  ·  {'  ·  '.join(MEMBERS)}  ·  {CONTACT}",
                color=INK, fontsize=13, ha="center", va="center")

    slides = [s_title, s_intro, s_task, s_model, s_obs, s_net,
              s_recovery, s_calib, s_ppc, s_compare, s_chaos, s_tldr]

    total = len(slides)
    with PdfPages(OUT_PDF) as pdf:
        for i, slide_fn in enumerate(slides, start=1):
            fig, bg = new_slide()
            slide_fn(fig, bg)
            if i > 1:
                footer(bg, i, total)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            fig.savefig(PREVIEW_DIR / f"slide_{i:02d}.png",
                        facecolor=fig.get_facecolor(), dpi=110)
            plt.close(fig)

    print(f"saved {OUT_PDF} ({total} slides)")
    print(f"previews in {PREVIEW_DIR}")


if __name__ == "__main__":
    build()
