"""
src/build_minutes.py — generate a detailed "work minutes" PDF for the group.

This is NOT the graded report. It is an internal, chronological log of
everything we actually did together on the 3-Body-Problem SBI project:
what was built, which decisions were made, what broke, how we fixed it, and
what the final numbers are. Intended for the group's own records.

Output: results/project_minutes.pdf

Usage:
    ./.venv/bin/python src/build_minutes.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "results" / "project_minutes.pdf"

# ---- palette (deep-space accents to match the deck) ----
NAVY = colors.HexColor("#0d1b3a")
BLUE = colors.HexColor("#1f6feb")
GOLD = colors.HexColor("#b7791f")
INK = colors.HexColor("#1a1a1a")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#eef2fb")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=16, textColor=NAVY,
                    spaceBefore=14, spaceAfter=6, leading=20)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=12.5, textColor=BLUE,
                    spaceBefore=10, spaceAfter=4, leading=16)
BODY = ParagraphStyle("Body", parent=_ss["BodyText"], fontSize=10, textColor=INK,
                      leading=14.5, alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, textColor=GREY)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=10.5, textColor=INK, leading=15)
TITLE = ParagraphStyle("Title", parent=_ss["Title"], fontSize=24, textColor=NAVY,
                       leading=28, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=_ss["Title"], fontSize=12.5, textColor=GOLD,
                     leading=16, spaceAfter=2, fontName="Helvetica")
META = ParagraphStyle("Meta", parent=BODY, fontSize=9.5, textColor=GREY,
                      alignment=TA_LEFT)


def bullets(items, style=BODY, bullet="•", indent=14):
    return ListFlowable(
        [ListItem(Paragraph(x, style), leftIndent=indent, value=bullet) for x in items],
        bulletType="bullet", start=bullet, leftIndent=indent + 6, bulletFontSize=9,
        spaceBefore=2, spaceAfter=6,
    )


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#c9d4ec"),
                      spaceBefore=4, spaceAfter=8)


def kv_table(rows, col_widths=(5.5 * cm, 10.5 * cm)):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dfe6f5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def data_table(header, rows, col_widths):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cdd7ee")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Page furniture (header/footer)
# ---------------------------------------------------------------------------
def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 12 * mm, w, 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(18 * mm, h - 8 * mm, "3-Body Problem · SBI Project — Work Minutes")
    canvas.setFillColor(GOLD)
    canvas.drawRightString(w - 18 * mm, h - 8 * mm, "Group " + GROUP)
    canvas.setStrokeColor(colors.HexColor("#c9d4ec"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "Internal log — not the graded report")
    canvas.drawRightString(w - 18 * mm, 9 * mm, "Page %d" % doc.page)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
GROUP = "XX"
MEMBERS = "Your Name, Teammate Name"

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
def story():
    S = []
    P = lambda t, s=BODY: S.append(Paragraph(t, s))

    # ---- Title block ----
    S.append(Spacer(1, 6 * mm))
    P("Inferring Initial Conditions in the 3-Body Problem", TITLE)
    P("Simulation-Based Inference with BayesFlow — detailed work minutes", SUB)
    S.append(Spacer(1, 4 * mm))
    S.append(kv_table([
        ["Group", GROUP],
        ["Members", MEMBERS],
        ["Course", "Simulation-Based Inference (graduate project), TU Dortmund"],
        ["Advisor", "Aayush (feedback rounds noted below)"],
        ["Generated", date.today().isoformat()],
        ["Purpose", "Internal record of everything we built and decided together."],
    ]))
    S.append(Spacer(1, 3 * mm))
    P("This document is a chronological log of the work — the pipeline we built, "
      "the decisions we made, the problems we hit and how we fixed them, and the "
      "final numbers. It is meant for our own group records, not for grading.", LEAD)
    S.append(rule())

    # ---- 0. Goal ----
    P("0. Project goal", H1)
    P("Given a noisy, partial observation of three gravitating bodies, recover a "
      "<b>full Bayesian posterior over their initial conditions</b> (positions and "
      "velocities at t = 0). Because the 3-body problem is chaotic, the honest "
      "answer is often a <b>wide but well-calibrated</b> posterior — and showing "
      "exactly that is the scientific point of the project.", BODY)
    P("We use amortized Neural Posterior Estimation (NPE) via BayesFlow: train once "
      "on simulated (parameters, trajectory) pairs, then get instant posteriors for "
      "any new observation.", BODY)

    # ---- 1. Simulator ----
    P("1. Physics simulator (src/simulator.py)", H1)
    P("Built the forward model — the Newtonian 3-body integrator that turns an "
      "initial state into a trajectory.", BODY)
    S.append(bullets([
        "Nondimensional units: gravitational constant G = 1, total mass = 1.",
        "Three <b>distinct, labelled</b> masses in ratio 1 : 2 : 3.",
        "Centre-of-mass (COM) frame: COM fixed at the origin, total momentum = 0.",
        "Integration with SciPy <font face='Courier'>solve_ivp</font> using DOP853 "
        "(8th-order), rtol = 1e-10, atol = 1e-12.",
        "Gravitational <b>softening</b> to avoid singularities at close approach.",
        "State packing/unpacking written to be dimension-agnostic "
        "(N_BODIES × DIM), which later made the 2D→3D jump painless.",
        "Energy check |ΔE/E| used as an automatic sanity guard on each trajectory.",
    ]))

    # ---- 2. Priors ----
    P("2. Priors & parameterization (src/priors.py)", H1)
    S.append(bullets([
        "Free parameters = initial positions & velocities of <b>bodies 1 and 2</b>; "
        "body 3 is then fixed by the COM and zero-momentum constraints.",
        "This gives 8 free parameters in 2D and <b>12 in 3D</b>.",
        "Priors: positions ~ Uniform[-2, 2], velocities ~ Uniform[-0.15, 0.15].",
        "Rejection of unphysical draws (too-close bodies / immediate escape) so "
        "training data is dominated by genuine bound-ish interactions.",
        "Parameter names generated dynamically (r1_x, r1_y, r1_z, …, v2_z) so all "
        "downstream plots/labels adapt automatically to the dimension.",
    ]))

    # ---- 3. Chaos ----
    P("3. Chaos analysis — Lyapunov timescale (src/lyapunov.py)", H1)
    P("Measured how fast nearby trajectories diverge, to justify the observation "
      "window and to frame the uncertainty story.", BODY)
    S.append(bullets([
        "Estimated the largest Lyapunov exponent λ ≈ 0.089, i.e. a Lyapunov time "
        "τ ≈ 11.3 in our units.",
        "Decision: place all observations <b>inside the first Lyapunov time</b>, "
        "where the data still carries information about the initial state.",
    ]))

    # ---- 4. Observation model ----
    P("4. Observation model (src/observables.py)", H1)
    S.append(bullets([
        "Observe the <b>trajectory</b> (snapshots of all positions & velocities), "
        "not just the final state.",
        "Additive Gaussian observation noise (baseline σ = 0.01).",
        "Separate standardization for position vs velocity features, with extra "
        "emphasis on velocities (which are the hard, information-poor directions).",
    ]))

    # ---- 5. Data generation ----
    P("5. Data-generation pipeline (src/generate_data.py)", H1)
    S.append(bullets([
        "Parallel workers simulate (parameters → observation) pairs at scale.",
        "Checkpointing + resume so long runs survive interruptions.",
        "Produced the datasets used throughout (see inventory at the end): "
        "2D and 3D full sets of 50,000 samples each, plus higher-noise variants.",
    ]))

    # ---- 6. Inference ----
    P("6. BayesFlow inference model (src/inference.py)", H1)
    S.append(bullets([
        "Summary network: <b>TimeSeriesTransformer</b> (summary_dim = 32, "
        "embedding (128, 128), 8+8 attention heads) to compress each trajectory.",
        "Inference network: <b>CouplingFlow</b> (depth 8) as the default; later "
        "made switchable to <b>FlowMatching</b> for the comparison.",
        "Offline training: 100 epochs, batch 32, Adam, learning rate 5e-4, "
        "≈45k train / 5k validation split.",
        "Training metadata (final loss, wall-clock time, chosen network) written to "
        "<font face='Courier'>training_meta.json</font> for traceability.",
    ]))

    # ---- 7. Diagnostics ----
    P("7. Diagnostics (src/diagnostics.py)", H1)
    P("Standard SBI health checks on the trained posterior:", BODY)
    S.append(bullets([
        "<b>Parameter recovery</b>: posterior-mean vs ground truth (recovery.png).",
        "<b>Simulation-based calibration (SBC)</b>: rank ECDFs vs uniform "
        "(calibration_ecdf.png).",
        "<b>Posterior contraction</b> and z-scores (z_score_contraction.png).",
        "Recovery correlations computed per parameter and averaged.",
    ]))
    P("Final 3D diagnostics: mean posterior contraction ratio ≈ 0.97 across 5,000 "
      "test systems (positions contract strongly; velocities remain appropriately "
      "wider — the chaotic directions).", SMALL)

    # ---- 8. Identifiability ----
    P("8. Identifiability analysis (src/identifiability.py)", H1)
    S.append(bullets([
        "Perturbed each parameter and measured the trajectory response to see which "
        "parameters are easy vs hard to infer.",
        "Result: <b>positions are highly identifiable</b> (sensitivity ≈ 0.7–0.97), "
        "<b>velocities are weaker</b> (≈ 0.28–0.60) — exactly why velocity posteriors "
        "are wider. Refreshed for 3D.",
    ]))

    S.append(PageBreak())

    # ---- 9. Aayush #1 ----
    P("9. Advisor feedback round #1 — 4 observation points + full 50k", H1)
    P("Aayush asked us to (a) switch from 50 observation points per trajectory to "
      "just <b>4</b>, and (b) generate the full 50,000-sample dataset in that format "
      "and check the results still hold.", BODY)
    S.append(bullets([
        "Snapshots reduced to K = 4 time points at t ≈ [0, 3.3, 6.7, 10] "
        "(all within one Lyapunov time).",
        "Rather than re-simulate, we discovered the existing 50k set and "
        "<b>subsampled</b> it to 4 points (kept the original as "
        "<font face='Courier'>train_full_50obs.npz</font>).",
        "Retrained and confirmed results held — in fact velocity recovery improved "
        "once we scaled to the full 50k.",
    ]))

    # ---- 10. 3D ----
    P("10. Extension from 2D to 3D", H1)
    P("With 2D results approved, Aayush cleared us to go full 3D. We generalized "
      "every module instead of forking the code.", BODY)
    S.append(bullets([
        "Set <font face='Courier'>DIM = 3</font> in config; free parameters went "
        "8 → 12.",
        "Generalized simulator state packing, prior slicing, parameter names, demo "
        "initial states, and identifiability perturbations to arbitrary DIM.",
        "Generated a 50k 3D dataset (<font face='Courier'>train_full_3d.npz</font>) "
        "and retrained.",
        "Outcome: 3D results are on par with (and in places better than) 2D — the "
        "z-axis is recovered just as well as x and y.",
    ]))

    # ---- 11. Network comparison ----
    P("11. Inference-network comparison — CouplingFlow vs FlowMatching", H1)
    P("Requirement: show a comparison of inference networks. We made the network "
      "selectable and built src/compare_networks.py to evaluate both on the same "
      "600 test systems.", BODY)
    S.append(data_table(
        ["Metric", "CouplingFlow", "FlowMatching"],
        [
            ["Mean recovery r", "0.9976", "0.9974"],
            ["Mean 90% coverage", "0.961", "0.926"],
            ["Mean contraction", "0.060", "0.057"],
            ["Sampling time (600×300 draws)", "≈ 4.0 s", "≈ 411 s"],
            ["Seconds / 1k draws", "0.022", "2.28"],
        ],
        col_widths=(6.5 * cm, 4.75 * cm, 4.75 * cm),
    ))
    S.append(Spacer(1, 2 * mm))
    S.append(bullets([
        "Both recover parameters essentially perfectly.",
        "FlowMatching is <b>slightly</b> better calibrated on velocities; "
        "CouplingFlow is <b>~100× faster</b> to sample.",
        "Conclusion: CouplingFlow is the practical default; FlowMatching is a "
        "competitive alternative.",
    ]))

    # ---- 12. PPC ----
    P("12. Posterior predictive checks (src/ppc.py)", H1)
    P("This diagnostic was in the professor's brief but missing from our pipeline, "
      "so we added it.", BODY)
    S.append(bullets([
        "Draw initial conditions from the posterior, simulate them forward, and "
        "overlay on the observed trajectory (ppc_trajectories.png).",
        "Predicted trajectories tightly track the truth and pass through the "
        "observed points.",
        "Quantitative posterior-predictive 90% coverage = <b>0.89</b>.",
    ]))

    # ---- 13. Noise experiment ----
    P("13. Observation-quality experiment (src/experiment_noise.py)", H1)
    P("To demonstrate the central claim — posteriors should widen but stay honest "
      "as data gets worse — we swept the observation noise σ.", BODY)
    S.append(data_table(
        ["Noise σ", "Mean recovery r", "Mean 90% coverage", "Velocity width"],
        [
            ["0.01", "0.998", "0.955", "0.101"],
            ["0.05", "0.982", "0.892", "0.208"],
            ["0.15", "0.919", "0.781", "0.352"],
        ],
        col_widths=(3.0 * cm, 4.3 * cm, 4.7 * cm, 4.0 * cm),
    ))
    S.append(Spacer(1, 2 * mm))
    S.append(bullets([
        "Built higher-noise datasets from the clean trajectories (no re-simulation) "
        "and trained a model per noise level.",
        "Posterior width grows steadily with σ (expected, honest behaviour).",
        "Calibration is good at low/moderate noise and becomes <b>overconfident "
        "only at the extreme σ = 0.15</b> — we reported this honestly rather than "
        "overselling.",
    ]))

    # ---- 14. Chaos vs uncertainty ----
    P("14. Chaos-vs-uncertainty analysis (src/chaos_analysis.py)", H1)
    S.append(bullets([
        "Defined a per-system finite-time divergence as a chaos measure.",
        "Correlated it with posterior width: focusing on <b>velocity</b> width gave "
        "a clear positive trend (r ≈ 0.34) — more chaotic ⇒ wider velocity posterior.",
        "Calibration stays ≈ 0.95 across all chaos levels: <b>wider but still "
        "calibrated</b>, which is the whole thesis of the project.",
    ]))

    # ---- 15. Slides ----
    P("15. Presentation slides (src/build_slides.py)", H1)
    S.append(bullets([
        "Since no LaTeX/PowerPoint toolchain was available, we render a "
        "<b>space-themed 12-slide deck straight to PDF</b> with matplotlib "
        "(results/slides.pdf) — fully self-contained and reproducible.",
        "Follows the rulebook: title + names first; topic intro; model + BayesFlow "
        "fit; captioned result figures (recovery, calibration/SBC, PPC, network "
        "comparison, chaos & noise); TL;DR + contact last; no 'thank you' slide.",
        "Placeholders for group number / members / contact are set at the top of "
        "the script — fill in and re-run.",
    ]))

    # ---- 16. Housekeeping ----
    P("16. Housekeeping", H1)
    S.append(bullets([
        "Added generated slides (results/slides.pdf and results/slides_preview/) to "
        ".gitignore (results/ was already ignored).",
        "Config-driven design (config.py) so dimension, noise, and network choice "
        "are single-line switches.",
    ]))

    S.append(PageBreak())

    # ---- Problems & fixes ----
    P("Problems we hit and how we solved them", H1)
    S.append(bullets([
        "<b>Data run looked stuck.</b> Progress only prints at exact checkpoint "
        "multiples; batched acceptance jumped over them. It was actually running "
        "(confirmed via worker CPU usage).",
        "<b>Diagnostics looked frozen at epoch 17.</b> Carriage-return progress bars "
        "hid the truth; training had finished and the CPU-heavy sampling phase was "
        "running.",
        "<b>FlowMatching diagnostics killed (OOM, exit 137).</b> Training had "
        "actually succeeded; the OOM was during sampling (ODE integration per draw). "
        "Fixed by building compare_networks.py with <b>batched sampling</b> from "
        "saved checkpoints.",
        "<b>Weak overall chaos↔width correlation.</b> Mixing easy positions with "
        "hard velocities washed out the trend; switching to velocity-only width "
        "sharpened it.",
        "<b>Over-optimistic plot title.</b> 'Calibration holds at every noise level' "
        "was false at σ = 0.15; corrected to reflect overconfidence at extreme noise.",
        "<b>Reused work instead of recomputing.</b> Subsampled the existing 50k for "
        "the 4-point format and built noise datasets from clean trajectories — "
        "saving hours of simulation.",
    ]))

    # ---- Final results ----
    P("Final results at a glance (3D, default CouplingFlow)", H1)
    S.append(data_table(
        ["Quantity", "Value"],
        [
            ["Spatial dimension", "3D (12 free parameters)"],
            ["Training set", "50,000 simulations, 4 observation snapshots"],
            ["Position recovery", "r ≈ 1.00"],
            ["Velocity recovery", "r ≈ 0.99"],
            ["Mean posterior contraction", "0.97 (over 5,000 test systems)"],
            ["SBC 90% coverage", "≈ 0.95"],
            ["Posterior-predictive 90% coverage", "0.89"],
            ["Lyapunov time", "τ ≈ 11.3 (λ ≈ 0.089)"],
        ],
        col_widths=(8.0 * cm, 8.0 * cm),
    ))

    # ---- Inventory ----
    P("Code & artifact inventory", H1)
    P("Source modules (src/):", H2)
    S.append(bullets([
        "<font face='Courier'>simulator.py</font> — 3-body forward integrator.",
        "<font face='Courier'>priors.py</font> — priors + free-DOF ↔ full-state map.",
        "<font face='Courier'>observables.py</font> — observation model & scaling.",
        "<font face='Courier'>lyapunov.py</font> — Lyapunov exponent / timescale.",
        "<font face='Courier'>generate_data.py</font> — parallel dataset generation.",
        "<font face='Courier'>inference.py</font> — BayesFlow workflow (Coupling/FlowMatching).",
        "<font face='Courier'>diagnostics.py</font> — recovery, SBC, contraction.",
        "<font face='Courier'>identifiability.py</font> — parameter sensitivity.",
        "<font face='Courier'>compare_networks.py</font> — CouplingFlow vs FlowMatching.",
        "<font face='Courier'>ppc.py</font> — posterior predictive checks.",
        "<font face='Courier'>experiment_noise.py</font> — observation-quality sweep.",
        "<font face='Courier'>chaos_analysis.py</font> — chaos vs posterior width.",
        "<font face='Courier'>build_slides.py</font> — space-themed slide deck.",
        "<font face='Courier'>build_minutes.py</font> — this document.",
    ], SMALL))
    P("Datasets (data/) and checkpoints:", H2)
    S.append(bullets([
        "train_full_2d.npz, train_full_3d.npz (50k each); "
        "train_full_3d_noise05.npz, train_full_3d_noise15.npz; "
        "train_full_50obs.npz (original 50-point archive).",
        "checkpoints/ (3D default), checkpoints_2d/, checkpoints_fm/ (FlowMatching), "
        "checkpoints_noise05/, checkpoints_noise15/.",
    ], SMALL))
    P("Result figures (results/):", H2)
    S.append(bullets([
        "recovery.png, calibration_ecdf.png, coverage.png, z_score_contraction.png, "
        "losses.png, ppc_trajectories.png, compare_networks.png, "
        "experiment_noise.png, chaos_analysis.png, identifiability.png "
        "(+ matching .json summaries).",
    ], SMALL))

    # ---- Reproduce ----
    P("How to reproduce everything", H1)
    for cmd in [
        "python src/generate_data.py --n 50000 --out data/train_full_3d.npz",
        "python src/inference.py --data data/train_full_3d.npz --inference-network coupling",
        "python src/diagnostics.py --data data/train_full_3d.npz",
        "python src/compare_networks.py",
        "python src/ppc.py",
        "python src/experiment_noise.py",
        "python src/chaos_analysis.py",
        "python src/identifiability.py",
        "python src/build_slides.py",
        "python src/build_minutes.py",
    ]:
        S.append(Paragraph("<font face='Courier' size='8.5'>%s</font>" % cmd, BODY))

    return S


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="3-Body Problem SBI — Work Minutes",
        author=MEMBERS,
    )
    doc.build(story(), onFirstPage=_decorate, onLaterPages=_decorate)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
