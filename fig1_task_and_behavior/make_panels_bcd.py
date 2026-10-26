#!/usr/bin/env python3
"""
Figure 1 panels B, C, D -- now entirely Python. Previously B/C came from
run_make_figure.m (MATLAB/Octave) and D from its own separate script,
make_panel_d_final.py, which additionally extracted its own one-off
colors/fonts from the existing task_and_behavior.pdf composite rather
than using the shared style. Both are superseded by this one script, so
all three panels share one data load, one set of bin edges, and one
palette (shared/pub_style.py) -- removing the MATLAB dependency for
Fig. 1 entirely and the drift between D's colors and B/C's.

  B  Psychometric function: p(correct) vs |orientation|, split by
     contrast (true two-level contrast_id band). No-intercept logistic
     fit: |orientation|=0 is a genuinely ambiguous stimulus, so forcing
     p=0.5 there is a physically motivated constraint, not just a
     convenience.
  C  Confidence: p(high confidence) vs |orientation|, split by contrast
     (true two-level contrast_id band). Logistic fit WITH intercept --
     confidence at |ori|=0 isn't constrained to any particular value.
  D  Psychometric function split by BOTH reported confidence (color:
     pub_style.CONF_HIGH/CONF_LOW) and contrast (line style: solid high /
     dashed low contrast). No-intercept fit per (confidence, contrast)
     cell, same rationale as B.

  Contrast split: data/data_behav.mat now carries a `contrast_id` field
  (the trial matrix's OWN categorical contrast-level index -- the same
  field fig3_cross_decoder/fig4_peb_pools/si_stimulus_difficulty_and_cv_
  controls/fig6_7_S1_S2_behavioral_models use directly, added here by
  row-position alignment with fig6_7_S1_S2_behavioral_models/data/
  real_behavior_full.csv; see add_contrast_id_to_data_behav.py).
  contrast_id==1 is the high-contrast band (raw contrast ~1.13-1.27,
  65.3% of trials) and contrast_id==2 is low (~0.44-0.78). An earlier
  version of this script instead used a per-animal-blind MEDIAN split of
  the raw, near-continuous "contrast" column pooled across both animals:
  that is NOT equivalent to this categorical split -- the median falls
  just inside the high-contrast band rather than in the gap between the
  two bands, misclassifying a meaningful fraction of trials. Fixed after
  an external reproducibility check caught the mismatch.

Replicates the retired run_make_figure.m's logfit/logfit_noint/bin_data
logic exactly (same ~5 quantile bin edges pooled across all trials, same
choice of no-intercept vs intercept per panel, same min-n=3 per-bin
filter, same population-SEM, i.e. ddof=0) so B/C's numbers match what the
MATLAB script produced. D is unchanged in substance from the old
make_panel_d_final.py -- only its colors/fonts moved to pub_style.

Each panel is saved both as its own PDF, sized to the same 325x275 pt
slot make_panel_d_final.py targeted (so each can drop into the existing
hand-assembled task_and_behavior.pdf layout in place of the old MATLAB
renders / old panel D), and all three are saved together as one row in a
combined preview PNG/PDF for quick visual QA -- that combined file is a
preview, not a compositing-ready replacement for the hand-assembled
figure (panel A is still a separate hand-drawn schematic with no source
file, see README).

Data: data/data_behav.mat (same file behind every Fig. 1 panel).
"""
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import statsmodels.api as sm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402

PS.apply_rcparams()

# Panel A (hand-drawn, Affinity Designer) sits directly above B/C/D in the
# final composite and uses 30pt Arial-Bold lettering with generously-sized
# labels. Arial itself isn't installed in this environment, so pub_style's
# shared font list (Helvetica/Arial/DejaVu Sans) silently falls back to
# DejaVu Sans here -- visibly different from Panel A once placed side by
# side. Liberation Sans is metric-compatible with, and visually close to,
# Arial, and IS installed -- prefer it for this figure only (not a
# pub_style.py-wide change, since no other figure was flagged for this).
mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]

# pub_style's shared sizes (FS_LBL=13/FS_TICK=11/FS_LEG=11/FS_TTL=13/
# FS_PANEL=17) are tuned for the dense, many-panel figures elsewhere in the
# paper (e.g. the 8-panel-per-row Figs. 6/7/S1/S2). Fig. 1's B/C/D are only
# three panels, placed at ~1.13x their native 325x275pt size once dropped
# into the composite (see the composite-build step) directly beside Panel
# A's much larger lettering -- pub_style's sizes read as visibly smaller
# there. Bumped locally, for this figure only:
FS_LBL = 16
FS_TICK = 14
FS_LEG = 13
FS_TTL = 18
# Panel A's own letter is Arial-Bold at 30pt; B/C/D are placed at ~1.13x
# scale in the composite (see composite-build script's fit-by-width factor,
# (column_width - 2*margin)/325 =~ 1.128), so 27pt here lands at ~30.5pt in
# the final page -- matching Panel A rather than pub_style's shared 17pt
# (which would land at ~19pt, visibly smaller than A).
FS_PANEL = 27

HERE = Path(__file__).resolve().parent
DATA = str(HERE / "data" / "data_behav.mat")
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

N_BINS = 5
# Same physical size make_panel_d_final.py targeted for the composite's
# panel-D slot; B and C now match it exactly too.
SLOT_W_PT, SLOT_H_PT = 325, 275

SPINE_OFFSET = 6  # pt -- detach the kept left/bottom spines from the corner
# they'd otherwise share (matches Figs. 3 & 4's make_pub_figure.py /
# make_figure4_composite.py; requested for Fig. 1 B-D specifically, not a
# pub_style-wide change).


def _offset_spines(ax, amount=SPINE_OFFSET):
    ax.spines["left"].set_position(("outward", amount))
    ax.spines["bottom"].set_position(("outward", amount))


def logfit(x, y):
    """Logistic regression WITH intercept: p = sigmoid(a + b*x)."""
    res = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
    a, b = res.params
    return float(a), float(b)


def logfit_noint(x, y):
    """Logistic regression, no intercept: p = sigmoid(b*x), b only --
    forces p=0.5 at x=0 (see module docstring)."""
    res = sm.GLM(y, x[:, None], family=sm.families.Binomial()).fit()
    return 0.0, float(res.params[0])


def bin_data(x, y, edges, min_n=3):
    n = len(edges) - 1
    cx, mn, se = [], [], []
    for i in range(n):
        m = (x >= edges[i]) & (x < edges[i + 1]) if i < n - 1 else x >= edges[i]
        if m.sum() < min_n:
            continue
        vals = y[m]
        cx.append(x[m].mean())
        mn.append(vals.mean())
        se.append(vals.std(ddof=0) / np.sqrt(m.sum()))
    return np.array(cx), np.array(mn), np.array(se)


def sigmoid(a, b, x):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def style_axes(ax, letter):
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("|Orientation| (°)", fontsize=FS_LBL)  # match B/C/D's
    # own prior convention (both run_make_figure.m and the old panel-D script
    # used the degree symbol here; Fig. 6/7/S1/S2's "(deg)" is that family's
    # own separate, pre-existing convention and is left as-is)
    ax.tick_params(direction="out", labelsize=FS_TICK, length=4)
    PS.despine(ax)
    _offset_spines(ax)
    panel_letter_big(ax, letter)


def panel_letter_big(ax, letter, x=-0.20, y=1.12):
    """Same placement logic as pub_style.panel_letter, but at this figure's
    own, locally-larger FS_PANEL (see module-level comment above) rather
    than pub_style's shared size. x nudged slightly further left than
    pub_style's default (-0.18) to keep the larger glyph's left edge clear
    of the y-axis tick labels."""
    ax.text(x, y, letter, transform=ax.transAxes, clip_on=False,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")


def new_slot_fig():
    fig = plt.figure(figsize=(SLOT_W_PT / 72, SLOT_H_PT / 72))
    # Height capped at 0.62 (not 0.70) so PS.panel_letter's default y=1.12
    # offset lands within the page instead of being clipped off the top
    # edge of this tightly-cropped 325x275pt slot (bug found by checking
    # the saved PDF's actual text content -- the letter was silently
    # missing from earlier renders of this script).
    ax = fig.add_axes([0.21, 0.19, 0.76, 0.62])
    return fig, ax


def draw_contrast_panel(ax, abs_ori, y, edges, lo, hi, params_lo, params_hi, x_fit, legend=False):
    for mask, col, lab, params in ((lo, PS.CONTRAST_LO, "Low contrast", params_lo),
                                    (hi, PS.CONTRAST_HI, "High contrast", params_hi)):
        a, b = params
        ax.plot(x_fit, sigmoid(a, b, x_fit), "-", color=col, lw=2, zorder=2)
        cx, mn, se = bin_data(abs_ori[mask], y[mask], edges)
        # clip_on=False: the lowest |orientation| bin sits right at x=0 --
        # the left edge of the data box -- so a clipped marker there would
        # be chopped in half. That was invisible while the left spine sat
        # flush against x=0 (the marker's hard edge lined up with the
        # spine), but once the spine was detached ("outward", see
        # style_axes/_offset_spines) the gap between the data box edge and
        # the spine made the clipped half-circle obviously cut off.
        ax.errorbar(cx, mn, yerr=se, fmt="o", color=col, ecolor=col,
                    markerfacecolor=col, markeredgecolor="w", markersize=9,
                    elinewidth=1.2, capsize=3.5, zorder=3, label=lab,
                    clip_on=False)
    if legend:
        ax.legend(frameon=False, loc="lower right", fontsize=FS_LEG,
                  handlelength=1.6, labelspacing=0.3)


def main():
    d = sio.loadmat(DATA, squeeze_me=True, struct_as_record=False)
    abs_ori = np.abs(np.asarray(d["orientation"], dtype=float))
    contrast = np.asarray(d["contrast"], dtype=float)
    contrast_id = np.asarray(d["contrast_id"], dtype=float)
    correct = np.asarray(d["correct"], dtype=float)
    conf = np.asarray(d["confidence"], dtype=float)
    n_trials = len(abs_ori)

    # True categorical contrast band (see module docstring) -- NOT a median
    # split of the raw, near-continuous `contrast` column.
    hi = contrast_id == 1
    lo = contrast_id == 2
    print(f"n = {n_trials} trials; true contrast bands (contrast_id)  "
          f"low n={lo.sum()}  high n={hi.sum()}  "
          f"(mean raw contrast: low={contrast[lo].mean():.3f}  high={contrast[hi].mean():.3f})")

    edges = np.quantile(abs_ori, np.linspace(0, 1, N_BINS + 1))
    edges[0] = 0.0
    edges[-1] = np.inf
    x_fit = np.linspace(0, abs_ori.max(), 300)

    # --- Panel B: psychometric, split by contrast ---------------------------
    params_b = {"lo": logfit_noint(abs_ori[lo], correct[lo]),
                "hi": logfit_noint(abs_ori[hi], correct[hi])}
    print(f"Psychometric  -- low:  a=0.000  b={params_b['lo'][1]:.4f}")
    print(f"Psychometric  -- high: a=0.000  b={params_b['hi'][1]:.4f}")

    figB, axB = new_slot_fig()
    draw_contrast_panel(axB, abs_ori, correct, edges, lo, hi, params_b["lo"], params_b["hi"],
                        x_fit, legend=True)
    axB.axhline(0.5, ls="--", color="0.7", lw=0.8, zorder=1)
    axB.set_ylim(0.4, 1.04)
    axB.set_ylabel("p(correct)", fontsize=FS_LBL)
    axB.set_title("Psychometric function", fontsize=FS_TTL, pad=6)
    style_axes(axB, "B")
    figB.savefig(FIGDIR / "panelB_psychometric.pdf")
    figB.savefig(FIGDIR / "panelB_psychometric.png", dpi=200)

    # --- Panel C: confidence, split by contrast ------------------------------
    params_c = {"lo": logfit(abs_ori[lo], conf[lo]),
                "hi": logfit(abs_ori[hi], conf[hi])}
    print(f"Confidence    -- low:  a={params_c['lo'][0]:.3f}  b={params_c['lo'][1]:.4f}")
    print(f"Confidence    -- high: a={params_c['hi'][0]:.3f}  b={params_c['hi'][1]:.4f}")

    figC, axC = new_slot_fig()
    draw_contrast_panel(axC, abs_ori, conf, edges, lo, hi, params_c["lo"], params_c["hi"], x_fit)
    axC.set_ylim(0, 1.04)
    axC.set_ylabel("p(high confidence)", fontsize=FS_LBL)
    axC.set_title("Confidence", fontsize=FS_TTL, pad=6)
    style_axes(axC, "C")
    figC.savefig(FIGDIR / "panelC_confidence.pdf")
    figC.savefig(FIGDIR / "panelC_confidence.png", dpi=200)

    # --- Panel D: psychometric, split by confidence (color) AND contrast
    # (line style) ------------------------------------------------------------
    groups = {
        ("high", "high"): hi & (conf == 1),
        ("low", "high"): lo & (conf == 1),
        ("high", "low"): hi & (conf == 0),
        ("low", "low"): lo & (conf == 0),
    }
    d_params = {key: logfit_noint(abs_ori[m], correct[m]) for key, m in groups.items()}
    for key, (a, b) in d_params.items():
        print(f"Psychometric (conf x contrast) -- conf={key[1]:>4s} contrast={key[0]:>4s}: "
              f"a=0.000  b={b:.4f}")

    style = {
        ("high", "high"): dict(color=PS.CONF_HIGH, ls="-"),
        ("low", "high"): dict(color=PS.CONF_HIGH, ls="--"),
        ("high", "low"): dict(color=PS.CONF_LOW, ls="-"),
        ("low", "low"): dict(color=PS.CONF_LOW, ls="--"),
    }
    figD, axD = new_slot_fig()
    axD.axhline(0.5, ls="--", color="0.7", lw=0.8, zorder=1)
    for key, m in groups.items():
        a, b = d_params[key]
        sty = style[key]
        axD.plot(x_fit, sigmoid(a, b, x_fit), lw=2, zorder=2, **sty)
        cx, mn, se = bin_data(abs_ori[m], correct[m], edges)
        mfc = sty["color"] if key[0] == "high" else "white"
        axD.errorbar(cx, mn, yerr=se, fmt="o", color=sty["color"],
                     markerfacecolor=mfc, markeredgecolor=sty["color"],
                     markeredgewidth=1.2, markersize=5.5, elinewidth=1.2,
                     capsize=3, ls="none", zorder=3, clip_on=False)
    axD.set_ylim(0.4, 1.02)
    axD.set_ylabel("p(correct)", fontsize=FS_LBL)
    axD.set_title("Confidence split", fontsize=FS_TTL, pad=6)
    style_axes(axD, "D")

    color_handles = [
        Line2D([], [], color=PS.CONF_HIGH, lw=2, marker="o", markersize=5,
               markerfacecolor=PS.CONF_HIGH, markeredgecolor=PS.CONF_HIGH,
               label="High confidence"),
        Line2D([], [], color=PS.CONF_LOW, lw=2, marker="o", markersize=5,
               markerfacecolor=PS.CONF_LOW, markeredgecolor=PS.CONF_LOW,
               label="Low confidence"),
    ]
    style_handles = [
        Line2D([], [], color="0.2", lw=1.8, ls="-", label="High contrast"),
        Line2D([], [], color="0.2", lw=1.8, ls="--", label="Low contrast"),
    ]
    leg1 = axD.legend(handles=color_handles, loc="lower right", frameon=False,
                      fontsize=FS_LEG, handlelength=1.6, labelspacing=0.3,
                      bbox_to_anchor=(0.98, 0.50), borderaxespad=0.0)
    axD.add_artist(leg1)
    axD.legend(handles=style_handles, loc="lower right", frameon=False,
              fontsize=FS_LEG, handlelength=1.6, labelspacing=0.3,
              bbox_to_anchor=(0.98, 0.22), borderaxespad=0.0)

    figD.savefig(FIGDIR / "panelD_final.pdf")
    figD.savefig(FIGDIR / "panelD_final.png", dpi=200)

    # --- Combined 3-up preview (QA only, not a compositing-ready file) -------
    prev_fig, prev_axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    draw_contrast_panel(prev_axes[0], abs_ori, correct, edges, lo, hi,
                        params_b["lo"], params_b["hi"], x_fit, legend=True)
    prev_axes[0].axhline(0.5, ls="--", color="0.7", lw=0.8, zorder=1)
    prev_axes[0].set_ylim(0.4, 1.04)
    prev_axes[0].set_ylabel("p(correct)", fontsize=FS_LBL)
    prev_axes[0].set_title("B: Psychometric function", fontsize=FS_TTL, pad=6)

    draw_contrast_panel(prev_axes[1], abs_ori, conf, edges, lo, hi,
                        params_c["lo"], params_c["hi"], x_fit)
    prev_axes[1].set_ylim(0, 1.04)
    prev_axes[1].set_ylabel("p(high confidence)", fontsize=FS_LBL)
    prev_axes[1].set_title("C: Confidence", fontsize=FS_TTL, pad=6)

    ax = prev_axes[2]
    ax.axhline(0.5, ls="--", color="0.7", lw=0.8, zorder=1)
    for key, m in groups.items():
        a, b = d_params[key]
        sty = style[key]
        ax.plot(x_fit, sigmoid(a, b, x_fit), lw=2, zorder=2, **sty)
        cx, mn, se = bin_data(abs_ori[m], correct[m], edges)
        mfc = sty["color"] if key[0] == "high" else "white"
        ax.errorbar(cx, mn, yerr=se, fmt="o", color=sty["color"],
                    markerfacecolor=mfc, markeredgecolor=sty["color"],
                    markeredgewidth=1.2, markersize=5.5, elinewidth=1.2,
                    capsize=3, ls="none", zorder=3)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("p(correct)", fontsize=FS_LBL)
    ax.set_title("D: Confidence split", fontsize=FS_TTL, pad=6)

    for ax, letter in zip(prev_axes, "BCD"):
        style_axes(ax, letter)
    prev_fig.tight_layout(w_pad=2.0)
    prev_fig.savefig(FIGDIR / "panels_BCD_preview.pdf", bbox_inches="tight")
    prev_fig.savefig(FIGDIR / "panels_BCD_preview.png", dpi=200, bbox_inches="tight")

    print(f"\nSaved {FIGDIR / 'panelB_psychometric.pdf'}")
    print(f"Saved {FIGDIR / 'panelC_confidence.pdf'}")
    print(f"Saved {FIGDIR / 'panelD_final.pdf'}")
    print(f"Saved {FIGDIR / 'panels_BCD_preview.pdf'} (QA preview only)")


if __name__ == "__main__":
    main()
