#!/usr/bin/env python3
"""Composite Figure 4: the positive-evidence bias in the two sensory pools.

    PANEL A  Pool-fit weights in the (a, b) plane, both targets overlaid,
             with the three limiting-case read-outs marked: b = 0 (pure
             positive-evidence bias), b = -a (calibrated decision-variable
             read-out) and b = +a (pure magnitude/gain read-out). One point
             per session x choice pair, per target; stars are across-pair
             means. This is the model-based statement of the asymmetry.
    PANEL B  The same asymmetry without a model: mean z-scored activity
             difference in each pool for the confidence split and the
             accuracy split, across sessions (bars + SEM, no per-session
             dots -- panel A already shows the per-pair scatter).
    PANEL C  Time-resolved version, confidence split.
    PANEL D  Time-resolved version, accuracy split.

Panels C and D reuse the plotting and statistics code of make_figure1.py
verbatim; panel B is that script's bar panel with the session dots removed;
panel A replaces the former standalone peb_vs_me_pub.pdf figure.

DATA   data/figure1_activity.csv                                   (panel B)
       data/figure1_timecourse.npz                              (panels C,D)
       ../residual_axis_confidence/data/peb_vs_me.csv    (panel A, a/b conf)
       ../residual_axis_accuracy/data/peb_vs_me_accuracy.csv (panel A, acc)
OUTPUT figures/figure4_peb_composite.pdf
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))

import pub_style as PS  # noqa: E402
PS.apply_rcparams()

from stats_summary import sig_str  # noqa: E402
from make_figure1 import _psth_panel, _diff_traces, _sem  # noqa: E402

# ---- colours -------------------------------------------------------------
# Panel A distinguishes the two TARGETS (confidence vs. accuracy); panels
# B-D distinguish the two POOLS (chosen- vs. unchosen-preferring). Two
# different jobs, so two disjoint hue pairs -- green/salmon are reserved
# for the pools throughout, and never used for a target.
COL_CONF_TGT = PS.CONF_HIGH   # warm amber-red  -- confidence, as a target
COL_ACC_TGT = PS.CONF_LOW     # cool steel-blue -- accuracy, as a target
COL_CHOSEN = PS.CHOSEN
COL_UNCHOSEN = PS.UNCHOSEN
REF = "0.35"

# Panel A's across-pair mean "stars" sat in the densest part of each point
# cloud and, sharing the cloud's own (semi-transparent) hue, were easy to
# lose among the dots. Pulled to a clearer, fully-saturated red/blue --
# distinct from COL_CONF_TGT/COL_ACC_TGT -- with a white halo edge so the
# star reads as a distinct summary marker rather than just a bigger dot.
STAR_CONF = "#e8000b"
STAR_ACC = "#0a3fff"

SPINE_OFFSET = 6  # pt -- detach the kept left/bottom spines from the
# corner they'd otherwise share (requested for this figure only, not a
# pub_style-wide change). Applied here to panels A and B, and inside
# make_figure1.py's _psth_panel (shared by panels C and D).


def _offset_spines(ax, amount=SPINE_OFFSET):
    ax.spines["left"].set_position(("outward", amount))
    ax.spines["bottom"].set_position(("outward", amount))

# ---- paths ---------------------------------------------------------------
ACT_CSV = os.path.join(HERE, "data", "figure1_activity.csv")
TC_NPZ = os.path.join(HERE, "data", "figure1_timecourse.npz")
CONF_CSV = os.path.join(HERE, "data", "peb_vs_me.csv")
ACC_CSV = os.path.join(HERE, "data", "peb_vs_me_accuracy.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
PDF_OUT = os.path.join(FIG_DIR, "figure4_peb_composite.pdf")

# ---- data ----------------------------------------------------------------
df = pd.read_csv(ACT_CSV)
tc = dict(np.load(TC_NPZ))

_c = pd.read_csv(CONF_CSV)[["session", "choice", "a_conf", "b_conf"]].dropna()
_a = pd.read_csv(ACC_CSV)[["session", "choice", "a_acc", "b_acc"]].dropna()
pf = _c.merge(_a, on=["session", "choice"])
n_pf = len(pf)

# ==========================================================================
#  PANEL A -- pool-fit weights in the (a, b) plane
# ==========================================================================
LIM = (-3.0, 4.0)


def panel_a(ax):
    # limiting-case reference lines, directly labelled at their ends
    lo, hi = LIM
    ax.axhline(0.0, color=REF, lw=1.2, ls="-", zorder=1)
    ax.plot([lo, hi], [-lo, -hi], color=REF, lw=1.2, ls="--", zorder=1)
    ax.plot([lo, hi], [lo, hi], color=REF, lw=1.2, ls=":", zorder=1)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)

    for key_a, key_b, color, star_color, label in (
            ("a_conf", "b_conf", COL_CONF_TGT, STAR_CONF, "Confidence"),
            ("a_acc", "b_acc", COL_ACC_TGT, STAR_ACC, "Accuracy")):
        av, bv = pf[key_a].to_numpy(), pf[key_b].to_numpy()
        ax.scatter(av, bv, s=26, facecolor=color, edgecolor="0.15",
                   linewidth=0.4, alpha=0.75, zorder=3, label=label)
        ax.scatter([av.mean()], [bv.mean()], marker="*", s=380,
                   facecolor=star_color, edgecolor="white", linewidth=1.3,
                   zorder=5)

    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$a$ = weight on $E^{+}$ (chosen pool)", fontsize=PS.FS_LBL)
    ax.set_ylabel(r"$b$ = weight on $E^{-}$ (unchosen pool)", fontsize=PS.FS_LBL)
    ax.spines[["top", "right"]].set_visible(False)
    _offset_spines(ax)
    ax.tick_params(labelsize=PS.FS_TICK)

    # One legend, in the empty upper-left corner: the two targets and the
    # three limiting-case read-outs. Direct labels on the lines themselves
    # would land in the point cloud.
    handles = [
        Line2D([], [], ls="none", marker="o", markersize=7,
               markerfacecolor=COL_CONF_TGT, markeredgecolor="0.15",
               markeredgewidth=0.4, label="Confidence"),
        Line2D([], [], ls="none", marker="o", markersize=7,
               markerfacecolor=COL_ACC_TGT, markeredgecolor="0.15",
               markeredgewidth=0.4, label="Accuracy"),
        Line2D([], [], color=REF, lw=1.2, ls="-", label=r"$b=0$  (pure PEB)"),
        Line2D([], [], color=REF, lw=1.2, ls="--", label=r"$b=-a$  (DV read-out)"),
        Line2D([], [], color=REF, lw=1.2, ls=":", label=r"$b=+a$  (pure gain)"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                     fontsize=9.5, handletextpad=0.5, handlelength=1.7,
                     labelspacing=0.35, borderaxespad=0.2)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.3")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1.0)

    out = int(((pf[["a_conf", "a_acc"]] > LIM[1]).any(axis=1) |
               (pf[["a_conf", "a_acc"]] < LIM[0]).any(axis=1) |
               (pf[["b_conf", "b_acc"]] > LIM[1]).any(axis=1) |
               (pf[["b_conf", "b_acc"]] < LIM[0]).any(axis=1)).sum())
    return out


# ==========================================================================
#  PANEL B -- model-free pool summary (no per-session dots)
# ==========================================================================
BAR_W = 0.62
XP = {"chosen_conf": 0.0, "chosen_acc": 0.68,
      "unchosen_conf": 1.9, "unchosen_acc": 2.58}


def _bracket(ax, x1, x2, y, tick, text):
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color="0.2", lw=1.0, clip_on=False, zorder=4)
    ax.text((x1 + x2) / 2, y + tick, text, ha="center", va="bottom",
            fontsize=PS.FS_LEG, clip_on=False)


def panel_b(ax):
    n = len(df)
    series = {
        "chosen_conf": df.chosen_conf_diff.to_numpy(float),
        "chosen_acc": df.chosen_acc_diff.to_numpy(float),
        "unchosen_conf": df.unchosen_conf_diff.to_numpy(float),
        "unchosen_acc": df.unchosen_acc_diff.to_numpy(float),
    }
    color = {"chosen_conf": COL_CHOSEN, "chosen_acc": COL_CHOSEN,
             "unchosen_conf": COL_UNCHOSEN, "unchosen_acc": COL_UNCHOSEN}
    is_acc = {"chosen_conf": False, "chosen_acc": True,
              "unchosen_conf": False, "unchosen_acc": True}

    for key, vals in series.items():
        x, c = XP[key], color[key]
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)
        if is_acc[key]:
            ax.bar(x, m, width=BAR_W, facecolor="white", edgecolor=c,
                   linewidth=1.8, zorder=1)
        else:
            ax.bar(x, m, width=BAR_W, facecolor=c, edgecolor=c,
                   linewidth=1.8, alpha=0.85, zorder=1)
        ax.errorbar(x, m, yerr=sem, fmt="none", ecolor="0.15",
                    elinewidth=1.3, capsize=4, zorder=3)

    ax.axhline(0, color="0.15", lw=0.8, zorder=0)

    w_ch = stats.wilcoxon(series["chosen_conf"], series["chosen_acc"])
    w_uch = stats.wilcoxon(series["unchosen_conf"], series["unchosen_acc"])

    means = {k: v.mean() for k, v in series.items()}
    sems = {k: v.std(ddof=1) / np.sqrt(n) for k, v in series.items()}
    yr = (max(means.values()) + max(sems.values())) - \
         (min(means.values()) - max(sems.values()))
    tick = 0.055 * yr
    top_ch = max(means["chosen_conf"] + sems["chosen_conf"],
                 means["chosen_acc"] + sems["chosen_acc"])
    top_uch = max(means["unchosen_conf"] + sems["unchosen_conf"],
                  means["unchosen_acc"] + sems["unchosen_acc"])
    _bracket(ax, XP["chosen_conf"], XP["chosen_acc"], top_ch + 0.12 * yr,
             tick, sig_str(w_ch.pvalue))
    _bracket(ax, XP["unchosen_conf"], XP["unchosen_acc"], top_uch + 0.12 * yr,
             tick, sig_str(w_uch.pvalue))

    ax.set_xlim(XP["chosen_conf"] - 0.55, XP["unchosen_acc"] + 0.55)
    ax.set_xticks(list(XP.values()))
    ax.set_xticklabels(["Conf.", "Acc.", "Conf.", "Acc."], fontsize=PS.FS_TICK)
    ax.tick_params(axis="x", length=0)
    for xc, lab in (((XP["chosen_conf"] + XP["chosen_acc"]) / 2, "Chosen-pref"),
                    ((XP["unchosen_conf"] + XP["unchosen_acc"]) / 2, "Unchosen-pref")):
        ax.text(xc, -0.10, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=PS.FS_LBL)
    ax.set_ylabel("Mean activity difference\n(z-scored spk)", fontsize=PS.FS_LBL)
    ax.spines[["top", "right"]].set_visible(False)
    _offset_spines(ax)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.16 * (ymax - ymin))
    ax.legend(handles=[
        Patch(facecolor="0.45", edgecolor="0.45", alpha=0.85, label="Confidence split"),
        Patch(facecolor="white", edgecolor="0.45", linewidth=1.8, label="Accuracy split")],
        fontsize=PS.FS_LEG, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4, columnspacing=1.0)
    return w_ch, w_uch


# ==========================================================================
#  figure
# ==========================================================================
fig = plt.figure(figsize=(9.8, 9.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 0.88],
                      height_ratios=[1.0, 0.86],
                      left=0.09, right=0.975, bottom=0.085, top=0.93,
                      wspace=0.40, hspace=0.42)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

n_out = panel_a(axA)
w_ch, w_uch = panel_b(axB)


def _band_minmax(arr):
    m, s = arr.mean(axis=0), _sem(arr)
    return (m - s).min(), (m + s).max()


traces = [*_diff_traces(tc, "high", "low"), *_diff_traces(tc, "correct", "incorrect")]
bounds = [_band_minmax(a) for a in traces]
dmin, dmax = min(b[0] for b in bounds), max(b[1] for b in bounds)
dyr = dmax - dmin
psth_ylim = (dmin - 0.20 * dyr, dmax + 0.10 * dyr)

_psth_panel(axC, tc, "high", "low", "Confidence split", psth_ylim, show_legend=True)
_psth_panel(axD, tc, "correct", "incorrect", "Accuracy split", psth_ylim)
axC.set_ylabel("Baseline-corrected\nfiring rate (spk/s)", fontsize=PS.FS_LBL)

for ax, letter, xo in ((axA, "A", -0.16), (axB, "B", -0.20),
                       (axC, "C", -0.16), (axD, "D", -0.14)):
    PS.panel_letter(ax, letter, x=xo, y=1.04)

fig.savefig(PDF_OUT)
fig.savefig(PDF_OUT.replace(".pdf", ".png"), dpi=170)

# ---- console statistics --------------------------------------------------
print(f"Panel A: n = {n_pf} session x choice pairs per target; "
      f"{n_out} pair(s) fall outside the plotted range {LIM}")
for k in ["a_conf", "b_conf", "a_acc", "b_acc"]:
    v = pf[k]
    print(f"    {k:7s} mean {v.mean():+.3f}  median {v.median():+.3f}  "
          f"vs 0: p = {stats.wilcoxon(v).pvalue:.3g}")
print(f"    paired a (conf - acc): {(pf.a_conf - pf.a_acc).mean():+.3f}, "
      f"p = {stats.wilcoxon(pf.a_conf - pf.a_acc).pvalue:.3g}")
print(f"    paired b (conf - acc): {(pf.b_conf - pf.b_acc).mean():+.3f}, "
      f"p = {stats.wilcoxon(pf.b_conf - pf.b_acc).pvalue:.3g}")
print(f"Panel B: n = {len(df)} sessions; "
      f"chosen conf vs acc {sig_str(w_ch.pvalue)} (p = {w_ch.pvalue:.3g}); "
      f"unchosen conf vs acc {sig_str(w_uch.pvalue)} (p = {w_uch.pvalue:.3g})")
print(f"\nSaved: {PDF_OUT}")
