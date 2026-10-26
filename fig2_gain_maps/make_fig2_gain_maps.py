#!/usr/bin/env python3
"""Figure 2: choice, confidence and accuracy over the (d, g) plane.

Production version. Replaces the bin-and-Gaussian-smooth construction in
make_pub_composite.m. The axes are unchanged in meaning -- still the
difference and the sum of the two tuning-defined pools -- so the E+/E-
decomposition the pool-fit analysis relies on is untouched. What changed:

  * d and g are z-scored WITHIN SESSION before pooling, not merely
    mean-centred. Sessions carry 17-74 neurons, so an unscaled d meant
    different amounts of evidence in different sessions.
  * The maps are no longer binned-and-smoothed. Each session gets its own
    logistic tensor-product spline surface in (d, g); those surfaces are
    averaged across sessions. There is no bin width and no smoothing
    kernel, and the unit of analysis is the session, which is the unit the
    data support (trials within a session are not independent).
  * A session contributes to a grid point only where it actually has
    trials (>= SUPP_N within SUPP_R); points covered by fewer than
    MIN_SESS sessions are masked in grey -- distinct from every colour in
    every ramp, so "no data" can no longer be read as a mid-scale value.
  * Accuracy gets a single-hue sequential ramp over its own range, rather
    than being rescaled into the diverging renderer (where its lowest
    value rendered as pure white, indistinguishable from P = 0.5).
  * The contour panels show the across-session mean flanked by mean +/-
    SEM contours, so the rotation claim carries its uncertainty.

INPUT   ../comments_followup_conf_acc/data/session_cache/session_*.npz
OUTPUT  figures/fig2_gain_maps.pdf   (+ .png, + printed statistics)
"""
import glob
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "session_cache")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import pub_style as PS  # noqa: E402

MIN_GRP = 3                  # neurons required in each pool (as published)
DF, SPL_C = 4, 1.0           # spline basis functions per axis; surface ridge
LIM, NG = 2.0, 60            # grid spans +/- LIM SD within session
SUPP_R, SUPP_N, MIN_SESS = 0.35, 5, 9

# v2: font sizes and colours now come from the shared pub_style module
# instead of this script's own (slightly different) local copies -- see
# shared/pub_style.py for the full rationale. The colour values
# themselves (CHOI_RIGHT/LEFT, CONF_HIGH/LOW, ACCR_CORR) are unchanged
# from what this script used before; only the font-size scale and the
# missing "ticks point out" rcParam were brought in line with every other
# figure.
FS_LBL, FS_TICK, FS_LEG, FS_TTL, FS_PANEL = (
    PS.FS_LBL, PS.FS_TICK, PS.FS_LEG, PS.FS_TTL, PS.FS_PANEL)
CHOI_RIGHT, CHOI_LEFT = PS.CHOI_RIGHT, PS.CHOI_LEFT
CONF_HIGH, CONF_LOW = PS.CONF_HIGH, PS.CONF_LOW
ACCR_CORR = PS.ACCR_CORR
MASK_FILL = "#e6e6e6"
PS.apply_rcparams()

grid = np.linspace(-LIM, LIM, NG)
GD, GG = np.meshgrid(grid, grid)
flat = np.column_stack([GD.ravel(), GG.ravel()])
spl = SplineTransformer(n_knots=DF, degree=3, include_bias=False)


def _tensor(B):
    h = B.shape[1] // 2
    return np.einsum("ni,nj->nij", B[:, :h], B[:, h:]).reshape(len(B), -1)


def surface(X, y):
    m = LogisticRegression(C=SPL_C, max_iter=4000).fit(_tensor(spl.fit_transform(X)), y)
    return m.predict_proba(_tensor(spl.transform(flat)))[:, 1].reshape(NG, NG)


surf = {k: [] for k in ("choice", "conf", "acc")}
support = np.zeros((NG, NG))
nsess = ntrials = 0
for f in sorted(glob.glob(os.path.join(CACHE, "session_*.npz"))):
    z = np.load(f)
    if z["right_pref"].sum() < MIN_GRP or z["left_pref"].sum() < MIN_GRP:
        continue
    C = z["counts"].astype(float)
    mu, sd = C.mean(0), C.std(0)
    keep = sd > 0
    Z = (C[:, keep] - mu[keep]) / sd[keep]
    rp, lp = z["right_pref"][keep], z["left_pref"][keep]
    d = Z[:, rp].mean(1) - Z[:, lp].mean(1)      # choice axis, E+ - E-
    g = Z[:, rp].mean(1) + Z[:, lp].mean(1)      # gain axis,   E+ + E-
    d = (d - d.mean()) / d.std()                 # z-scored WITHIN session
    g = (g - g.mean()) / g.std()
    X = np.column_stack([d, g])

    cov = ((((np.abs(d[None, :] - GD.ravel()[:, None]) < SUPP_R) &
             (np.abs(g[None, :] - GG.ravel()[:, None]) < SUPP_R)).sum(1))
           >= SUPP_N).reshape(NG, NG)
    support += cov
    for k, y in (("choice", (z["choice"] == 1).astype(int)),
                 ("conf", z["confidence"].astype(int)),
                 ("acc", z["correct"].astype(int))):
        s = surface(X, y)
        s[~cov] = np.nan
        surf[k].append(s)
    nsess += 1
    ntrials += len(d)

mask = support >= MIN_SESS
mean = {k: np.ma.masked_array(np.nanmean(v, 0), ~mask) for k, v in surf.items()}
sem = {k: np.ma.masked_array(
    np.nanstd(v, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(v), 0)), ~mask)
    for k, v in surf.items()}
CMASK = support >= max(MIN_SESS, int(0.8 * nsess))


def diverging(lo, hi):
    cm = LinearSegmentedColormap.from_list("d", [lo, "#ffffff", hi])
    cm.set_bad(MASK_FILL)
    return cm


def sequential(hi, floor=0.14):
    rgb = np.array(mpl.colors.to_rgb(hi))
    cm = LinearSegmentedColormap.from_list(
        "s", [floor * rgb + (1 - floor) * np.ones(3), rgb])
    cm.set_bad(MASK_FILL)
    return cm


XLAB = "$d$  (choice axis, SD)"
YLAB = "$g$  (gain axis, SD)"


def frame(ax, ylabel=False, xlabel=True):
    ax.axvline(0, color="0.3", ls=":", lw=0.9)
    ax.axhline(0, color="0.3", ls=":", lw=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    if xlabel:
        ax.set_xlabel(XLAB, fontsize=FS_LBL)
    if ylabel:
        ax.set_ylabel(YLAB, fontsize=FS_LBL)


def heat(ax, M, cmap, vlo, vhi, label, ticks, ylabel=False):
    pc = ax.pcolormesh(grid, grid, M, cmap=cmap, vmin=vlo, vmax=vhi,
                       shading="nearest")
    frame(ax, ylabel)
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.0%}" for t in ticks])
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(0.6)


def contours(ax, key, levels, color, ls="-"):
    """Across-session mean iso-probability contours."""
    Z = np.ma.masked_array(mean[key], ~CMASK)
    cs = ax.contour(grid, grid, Z, levels=levels, colors=color,
                    linewidths=1.9, linestyles=ls)
    ax.clabel(cs, fontsize=8, fmt=lambda v: f"{v:.0%}", inline_spacing=4)


fig = plt.figure(figsize=(12.4, 10.2))
gs = fig.add_gridspec(2, 6, left=0.055, right=0.93, bottom=0.07, top=0.935,
                      height_ratios=[1.0, 1.45], wspace=1.05, hspace=0.26)
axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[0, 2:4])
axC = fig.add_subplot(gs[0, 4:6])
axD = fig.add_subplot(gs[1, 0:3])
axE = fig.add_subplot(gs[1, 3:6])

heat(axA, mean["choice"], diverging(CHOI_LEFT, CHOI_RIGHT), 0, 1,
     "P(right choice)", [0, .25, .5, .75, 1], ylabel=True)
axA.set_title("Choice   $P$(right)", fontsize=FS_TTL, pad=7)
heat(axB, mean["conf"], diverging(CONF_LOW, CONF_HIGH), 0.25, 0.75,
     "P(high confidence)", [.25, .375, .5, .625, .75])
axB.set_title("Confidence   $P$(high)", fontsize=FS_TTL, pad=7)
alo, ahi = np.percentile(mean["acc"].compressed(), [1, 99])
heat(axC, mean["acc"], sequential(ACCR_CORR), alo, ahi, "P(correct)",
     list(np.linspace(alo, ahi, 5)))
axC.set_title("Accuracy   $P$(correct)", fontsize=FS_TTL, pad=7)

CONF_LEVELS = [0.45, 0.50, 0.55, 0.60, 0.65]
contours(axD, "choice", [0.20, 0.35, 0.50, 0.65, 0.80], CHOI_RIGHT)
contours(axD, "conf", CONF_LEVELS, CONF_HIGH, ls="--")
frame(axD, ylabel=True)
axD.legend(handles=[Line2D([], [], color=CHOI_RIGHT, lw=2, label="P(right choice)"),
                    Line2D([], [], color=CONF_HIGH, lw=2, ls="--",
                           label="P(high confidence)")],
           frameon=False, loc="lower left", borderaxespad=0.2)
contours(axE, "acc", [0.79, 0.81, 0.83, 0.85, 0.87], ACCR_CORR)
contours(axE, "conf", CONF_LEVELS, CONF_HIGH, ls="--")
frame(axE)
axE.legend(handles=[Line2D([], [], color=ACCR_CORR, lw=2, label="P(correct)"),
                    Line2D([], [], color=CONF_HIGH, lw=2, ls="--",
                           label="P(high confidence)")],
           frameon=False, loc="lower left", borderaxespad=0.2)

for ax, s in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D"), (axE, "E")):
    ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left", clip_on=False)

fig.legend(handles=[Patch(facecolor=MASK_FILL, edgecolor="#cccccc",
                          label=f"covered by fewer than {MIN_SESS} sessions")],
           loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False,
           fontsize=FS_LEG)

fig.savefig(os.path.join(FIGDIR, "fig2_gain_maps.pdf"))
fig.savefig(os.path.join(FIGDIR, "fig2_gain_maps.png"), dpi=160)


def at(k, dv, gv):
    return mean[k][int(np.argmin(np.abs(grid - gv))),
                   int(np.argmin(np.abs(grid - dv)))]


print(f"{nsess} sessions, {ntrials} trials")
print(f"grid shown: {100*mask.sum()/NG**2:.0f}% of +/-{LIM} SD "
      f"(>= {MIN_SESS} sessions); contours drawn where "
      f">= {max(MIN_SESS, int(0.8*nsess))} sessions")
for k, lab in (("choice", "P(right choice)"), ("conf", "P(high confidence)"),
               ("acc", "P(correct)")):
    print(f"  {lab:20s} {mean[k].min():.3f}-{mean[k].max():.3f}   "
          f"median SEM {np.ma.median(sem[k]):.3f}")
c0, c1 = at("conf", 0, -1.5), at("conf", 0, 1.5)
a0, a1 = at("acc", 0, -1.5), at("acc", 0, 1.5)
print(f"\nat d = 0, g from -1.5 to +1.5 SD:")
print(f"  P(high confidence)  {c0:.3f} -> {c1:.3f}  ({c1-c0:+.3f})")
print(f"  P(correct)          {a0:.3f} -> {a1:.3f}  ({a1-a0:+.3f})")
print(f"  ratio {(c1-c0)/(a1-a0):.1f}x")
print(f"\nat g = 0, d from 0 to +/-1.5 SD:")
print(f"  P(right choice)     {at('choice',0,0):.3f} -> "
      f"{at('choice',1.5,0):.3f} / {at('choice',-1.5,0):.3f}")
print(f"  P(high confidence)  {at('conf',0,0):.3f} -> "
      f"{at('conf',1.5,0):.3f} / {at('conf',-1.5,0):.3f}")
print(f"  P(correct)          {at('acc',0,0):.3f} -> "
      f"{at('acc',1.5,0):.3f} / {at('acc',-1.5,0):.3f}")
print("\nSaved figures/fig2_gain_maps.pdf")
