#!/usr/bin/env python3
"""Publication figure for the choice-conditioned cross-decoder analysis.

Three-panel composite:

    PANEL A  Per (session x choice) AUC scatter: conf-trained decoder vs
             acc-trained decoder, BOTH SCORED AGAINST CONFIDENCE.
             Points above the unity line favor the confidence-trained
             decoder.
    PANEL B  Per (session x choice) AUC scatter: acc-trained decoder vs
             conf-trained decoder, BOTH SCORED AGAINST ACCURACY (the
             mirror image of Panel A). Axes are kept in the same
             orientation as Panel A (x = accuracy-trained AUC, y =
             confidence-trained AUC), so points BELOW the unity line
             favor the accuracy-trained decoder.
    PANEL C  Histogram of cos(w_conf, w_acc): cosine similarity between the
             conf-trained and acc-trained decoder weight vectors.
DATA   data/cross_decoder_pairs.csv  (written by run_cross_decoder.py;
       must include auc_conf_on_acc -- added alongside auc_acc_on_acc)
OUTPUT figures/cross_decoder_pub.pdf  (single page, vector)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----- shared style --------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402
PS.apply_rcparams()

# grayscale palette (kept from the original design -- this figure is
# deliberately monochrome; the pool colours belong to the pool figure)
DOT_FACE, DOT_EDGE = "0.45", "0.15"
BAR_FACE, BAR_EDGE = "0.72", "0.10"
LINE_K = "0.0"

SPINE_OFFSET = 6  # pt -- detach the kept left/bottom spines from the
# corner they'd otherwise share (requested for this figure only, not a
# pub_style-wide change).


def _offset_spines(ax, amount=SPINE_OFFSET):
    ax.spines["left"].set_position(("outward", amount))
    ax.spines["bottom"].set_position(("outward", amount))

# ----- paths -------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, "data", "cross_decoder_pairs.csv")
fig_dir = os.path.join(here, "figures")
os.makedirs(fig_dir, exist_ok=True)
pdf_out = os.path.join(fig_dir, "cross_decoder_pub.pdf")

# ----- data --------------------------------------------------------------
d = pd.read_csv(csv_path)
x = d["auc_acc_on_conf"].to_numpy()      # acc-trained decoder, scored on conf
y = d["auc_conf_on_conf"].to_numpy()     # conf-trained decoder, scored on conf
ok = ~(np.isnan(x) | np.isnan(y))
x, y = x[ok], y[ok]
n = x.size

has_mirror = "auc_conf_on_acc" in d.columns
if has_mirror:
    xm = d["auc_conf_on_acc"].to_numpy()   # conf-trained decoder, scored on acc
    ym = d["auc_acc_on_acc"].to_numpy()    # acc-trained decoder,  scored on acc
    okm = ~(np.isnan(xm) | np.isnan(ym))
    xm, ym = xm[okm], ym[okm]
    nm = xm.size
    diff_m = ym - xm
    p_m = wilcoxon(diff_m).pvalue

cs = d["cos_conf_acc"].to_numpy()
cs = cs[~np.isnan(cs)]

diff = y - x
p = wilcoxon(diff).pvalue

print(f"Panel A: n = {n} (session x choice) pairs; "
      f"mean AUC conf {y.mean():.3f} vs acc {x.mean():.3f}; "
      f"Delta = {diff.mean():+.3f}, Wilcoxon p = {p:.2e}")
if has_mirror:
    print(f"Panel B (mirror, scored on acc): n = {nm}; "
          f"mean AUC acc {ym.mean():.3f} vs conf {xm.mean():.3f}; "
          f"Delta = {diff_m.mean():+.3f}, Wilcoxon p = {p_m:.2e}")
print(f"Panel C: n = {cs.size}; cos mean {cs.mean():+.2f}, median {np.median(cs):+.2f}")
# ----- figure ------------------------------------------------------------
ncols = 3 if has_mirror else 2
fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.6))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.84, wspace=0.42)
axA = axes[0]
axB = axes[1] if has_mirror else None
axC = axes[-1]


def _auc_scatter(ax, xv, yv, xlab, ylab):
    lims = [min(xv.min(), yv.min()) - 0.02, max(xv.max(), yv.max()) + 0.02]
    ax.plot(lims, lims, ls="--", color=LINE_K, lw=1.0, zorder=1)
    ax.scatter(xv, yv, s=38, facecolor=DOT_FACE, edgecolor=DOT_EDGE,
               linewidth=0.5, alpha=0.85, zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlab, fontsize=PS.FS_LBL)
    ax.set_ylabel(ylab, fontsize=PS.FS_LBL)
    ax.tick_params(labelsize=PS.FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    _offset_spines(ax)


# --- Panel A: AUC scatter, scored on confidence ---
_auc_scatter(axA, x, y,
             "AUC, accuracy-trained decoder",
             "AUC, confidence-trained decoder")
axA.set_title("Scored against confidence", fontsize=PS.FS_TTL, pad=8)

# --- Panel B: AUC scatter, scored on accuracy (mirror of A) ---
if has_mirror:
    _auc_scatter(axB, ym, xm,
                 "AUC, accuracy-trained decoder",
                 "AUC, confidence-trained decoder")
    axB.set_title("Scored against accuracy", fontsize=PS.FS_TTL, pad=8)

# --- Panel C: cosine-similarity histogram ---
axC.hist(cs, bins=20, range=(-1, 1), facecolor=BAR_FACE,
         edgecolor=BAR_EDGE, linewidth=0.6)
zero_ln = axC.axvline(0.0, color=LINE_K, lw=1.0, ls="-")
mean_ln = axC.axvline(cs.mean(), color=LINE_K, lw=1.8, ls="--")
axC.axvline(1.0, color=LINE_K, lw=0.5, ls=":")
axC.axvline(-1.0, color=LINE_K, lw=0.5, ls=":")
axC.set_xlim(-1.05, 1.05)
axC.set_xlabel(r"cosine similarity, $\cos(w_{\mathrm{conf}}, w_{\mathrm{acc}})$",
               fontsize=PS.FS_LBL)
axC.set_ylabel(r"count (session $\times$ choice pairs)", fontsize=PS.FS_LBL)
axC.tick_params(labelsize=PS.FS_TICK)
axC.spines[["top", "right"]].set_visible(False)
_offset_spines(axC)
axC.legend([zero_ln, mean_ln], ["Orthogonal (0)", "Mean"],
           loc="upper left", frameon=False, fontsize=PS.FS_LEG)

# --- panel letters ---
letters = ["A", "B", "C"] if has_mirror else ["A", "B"]
for ax, letter in zip(axes, letters):
    PS.panel_letter(ax, letter)

fig.savefig(pdf_out)
fig.savefig(pdf_out.replace(".pdf", ".png"), dpi=170)
print(f"\nSaved: {pdf_out}")
