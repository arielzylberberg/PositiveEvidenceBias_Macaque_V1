#!/usr/bin/env python3
"""
For each free-parameter confidence model, one figure with 8 panels: 2 rows
(Ziggy, Friedrich) x 4 columns (accuracy - all trials split by CONTRAST;
confidence - correct trials only, split by contrast; confidence - error
trials only, split by contrast; accuracy - all trials, split by reported
CONFIDENCE), each panel showing REAL data (binned by |orientation| in true
degrees, Wilson 95% CI) with a fitted curve overlaid in the same panel --
the standard psychophysics model-fit presentation.

Columns 1-3 (contrast split) are identical across all model figures for
column 1 (it's the shared choice/SDT fit) and differ per model for columns
2-3 (the confidence read-out being tested). Column 4 (confidence split) is
NOT part of the SDT model comparison at all -- it's a model-free empirical
diagnostic (real accuracy vs. real reported confidence, mirroring
analysis_az/psychometric_confidence_figure/run_make_figure.m's own
"Psychometric split by confidence" panel), with a simple no-intercept
probit fit per confidence level (s=0 is now a genuine, physically
meaningful reference point -- true |orientation|=0 -- so forcing
P(correct)=0.5 there is well-motivated, unlike with the old z-scored axis).
It uses a DIFFERENT color pair from the contrast split (pub_style's
CONF_HIGH/CONF_LOW, not CONTRAST_HI/CONTRAST_LO) precisely so it can't be
mistaken for another contrast panel.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import fit_free_param_models as M

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402

FS_LBL, FS_TICK, FS_LEG, FS_TTL, FS_PANEL = (
    PS.FS_LBL, PS.FS_TICK, PS.FS_LEG, PS.FS_TTL, PS.FS_PANEL)
# v2: this script used to invent its own contrast colour pair (red/blue)
# and its own confidence colour pair (teal/orange) -- both now come from
# pub_style, matching Fig. 1/Fig. 2's CONTRAST_HI/LO and CONF_HIGH/LOW
# respectively instead of each figure having its own scheme for the same
# two dimensions. See shared/pub_style.py for the full rationale.
#
# CONTRAST_HI/LO and CONF_HIGH/LOW both appear in this 8-panel figure
# (different columns), and the validator (dataviz skill's
# validate_palette.js) puts the CONTRAST_HI vs CONF_LOW separation at
# only Delta-E 6.1 under protanopia/tritanopia -- the 6-8 "legal only
# with secondary encoding" band, not a clean pass. So the contrast split
# (columns 1-3) gets a secondary, non-color encoding on top of hue: solid
# line + filled marker for high contrast, dashed line + open marker for
# low contrast (see `panel()`'s `secondary_encoding` argument). The
# reported-confidence split (column 4) doesn't need this -- it's the only
# color-coded dimension in that panel.
C_HI, C_LO = PS.CONTRAST_HI, PS.CONTRAST_LO      # high / low contrast
C_HICONF, C_LOCONF = PS.CONF_HIGH, PS.CONF_LOW   # high / low reported confidence
PS.apply_rcparams()


def wilson(k, n, zc=1.96):
    p = k / n
    den = 1 + zc**2 / n
    c = (p + zc**2 / (2 * n)) / den
    h = zc * np.sqrt(p * (1 - p) / n + zc**2 / (4 * n**2)) / den
    return c - h, c + h


def binned_points(d, ycol, bin_edges, splitcol="hi", min_n=8):
    s = d["absori"].to_numpy(); split = d[splitcol].to_numpy(); y = d[ycol].to_numpy()
    bins = pd.cut(s, bins=bin_edges, include_lowest=True)
    centers = pd.Series(s).groupby(bins, observed=False).mean()
    out = {0: ([], [], [], []), 1: ([], [], [], [])}
    for h in (0, 1):
        for b in centers.index:
            sel = (bins == b) & (split == h)
            n = int(sel.sum())
            if n < min_n or np.isnan(centers[b]):
                continue
            k = y[sel].sum()
            lo, up = wilson(k, n)
            xs, ys, los, ups = out[h]
            xs.append(centers[b]); ys.append(k / n); los.append(lo); ups.append(up)
    return out


def truncate_curve_to_data(curve, points, margin=1.0):
    """Truncate a (gs, y) model curve to the |orientation| range actually
    covered by real data in `points` (a binned_points dict, both hi/lo
    splits combined), plus a small margin (degrees). Used for the
    error-only confidence panels, where the orientation grid extends well
    past the last bin with enough error trials to plot (min_n=8 in
    binned_points): past that point the curve is pure extrapolation from
    the fitted model, not constrained by any real data, and can swing
    sharply (e.g. a visible upturn at the far right of some panels) in a
    way that is easy to mistake for a real, data-supported feature. If
    `points` has no data at all (shouldn't happen here), the curve is
    returned unchanged."""
    xs_all = list(points[0][0]) + list(points[1][0])
    if not xs_all:
        return curve
    xmax = max(xs_all) + margin
    gs, y = curve
    gs = np.asarray(gs); y = np.asarray(y)
    keep = gs <= xmax
    return gs[keep], y[keep]


def fit_noint_probit(s, correct):
    """1-parameter probit through the origin: P(correct)=Phi(a*s). Now that
    s is real |orientation| in degrees, s=0 genuinely is an objectively
    ambiguous stimulus, so forcing P(correct)=0.5 there (no intercept) is
    the right constraint -- matching run_make_figure.m's own logfit_noint."""
    def negll(theta):
        a = theta[0]
        p = np.clip(norm.cdf(a * s), 1e-9, 1 - 1e-9)
        return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).sum()
    res = minimize(negll, x0=[0.1], method="Nelder-Mead",
                    options=dict(xatol=1e-8, fatol=1e-8))
    return res.x[0]


def combined_ylim(*sources, pad=0.06):
    """sources: a mix of `points` dicts (from binned_points) and (x, y) curve
    tuples. Returns (lo, hi) spanning all of them, with a little padding, so
    that paired panels (e.g. confidence correct-only vs. error-only) can be
    put on the same y-axis for direct visual comparison."""
    vals = []
    for src in sources:
        if isinstance(src, dict):
            for h in (0, 1):
                xs, ys, los, ups = src[h]
                vals.extend(los); vals.extend(ups)
        else:
            _, y = src
            vals.extend(np.asarray(y).tolist())
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    return max(0.0, lo - pad * span), min(1.0, hi + pad * span)


def panel(ax, points, curve_hi, curve_lo, ylabel, title,
          col_hi=C_HI, col_lo=C_LO, lab_hi="high contrast", lab_lo="low contrast",
          secondary_encoding=False):
    """secondary_encoding: when True, the "lo" series also gets a dashed
    line and an open (unfilled) marker, on top of its own hue, so the two
    series stay distinguishable even where the hue separation alone is
    only borderline under color-vision deficiency (see module docstring).
    Used for the contrast split; not needed for the confidence split,
    which is the only color-coded dimension in its panel."""
    ls_for = {1: "-", 0: "--" if secondary_encoding else "-"}
    mfc_for = {1: None, 0: "none" if secondary_encoding else None}
    for h, col, lab in ((1, col_hi, lab_hi), (0, col_lo, lab_lo)):
        xs, ys, los, ups = points[h]
        if xs:
            order = np.argsort(xs)
            xs, ys, los, ups = (np.array(a)[order] for a in (xs, ys, los, ups))
            yerr_lo = np.clip(ys - los, 0, None)
            yerr_hi = np.clip(ups - ys, 0, None)
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o", ms=5,
                        color=col, ecolor=col, elinewidth=1, capsize=2, zorder=3,
                        markerfacecolor=mfc_for[h], label=lab)
    gs_hi, y_hi = curve_hi
    gs_lo, y_lo = curve_lo
    ax.plot(gs_hi, y_hi, ls_for[1], color=col_hi, lw=1.8, zorder=2, alpha=0.85)
    ax.plot(gs_lo, y_lo, ls_for[0], color=col_lo, lw=1.8, zorder=2, alpha=0.85)
    ax.set_xlabel("|orientation| (deg)", fontsize=FS_LBL)
    ax.set_ylabel(ylabel, fontsize=FS_LBL)
    ax.set_title(title, fontsize=FS_TTL, pad=6)
    ax.spines[["top", "right"]].set_visible(False)


def make_model_figure(fits, data, model, out_prefix, bin_edges, suptitle=True):
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.6))
    for row, animal in enumerate(("Ziggy", "Friedrich")):
        fit = fits[animal]
        d = data[data.animal == animal]

        pts_acc = binned_points(d, "correct", bin_edges, splitcol="hi")
        curve_acc_hi = M.predict_accuracy_curve(fit, 1)
        curve_acc_lo = M.predict_accuracy_curve(fit, 0)
        panel(axes[row, 0], pts_acc, curve_acc_hi, curve_acc_lo,
              "P(correct)", "accuracy (all trials)",
              secondary_encoding=True)

        d_corr = d[d.correct == 1]
        pts_c = binned_points(d_corr, "confidence", bin_edges, splitcol="hi")
        curve_c_hi = M.predict_confidence_curve(fit, model, 1, 1)
        curve_c_lo = M.predict_confidence_curve(fit, model, 0, 1)
        panel(axes[row, 1], pts_c, curve_c_hi, curve_c_lo,
              "P(high confidence)", "confidence, correct only",
              secondary_encoding=True)

        d_err = d[d.correct == 0]
        pts_e = binned_points(d_err, "confidence", bin_edges, splitcol="hi")
        curve_e_hi = M.predict_confidence_curve(fit, model, 1, 0)
        curve_e_lo = M.predict_confidence_curve(fit, model, 0, 0)
        # Don't plot the model curve past the last |orientation| bin with
        # enough error trials to show real data (see truncate_curve_to_data
        # docstring) -- beyond that point the curve is unconstrained
        # extrapolation, not a fitted feature of the data.
        curve_e_hi = truncate_curve_to_data(curve_e_hi, pts_e)
        curve_e_lo = truncate_curve_to_data(curve_e_lo, pts_e)
        panel(axes[row, 2], pts_e, curve_e_hi, curve_e_lo,
              "P(high confidence)", "confidence, error only",
              secondary_encoding=True)

        # Same y-axis limits for the correct-only and error-only confidence
        # panels (this row) so the two are directly, visually comparable.
        ylo, yhi = combined_ylim(pts_c, pts_e, curve_c_hi, curve_c_lo, curve_e_hi, curve_e_lo)
        axes[row, 1].set_ylim(ylo, yhi)
        axes[row, 2].set_ylim(ylo, yhi)

        # Column 4: model-free empirical diagnostic -- real accuracy split by
        # reported CONFIDENCE (not contrast). Same across all model figures.
        pts_conf = binned_points(d, "correct", bin_edges, splitcol="confidence")
        s_all = d["absori"].to_numpy(); conf_all = d["confidence"].to_numpy()
        corr_all = d["correct"].to_numpy()
        a_hi = fit_noint_probit(s_all[conf_all == 1], corr_all[conf_all == 1])
        a_lo = fit_noint_probit(s_all[conf_all == 0], corr_all[conf_all == 0])
        gs = fit["grid_s"]
        curve_conf_hi = (gs, norm.cdf(a_hi * gs))
        curve_conf_lo = (gs, norm.cdf(a_lo * gs))
        panel(axes[row, 3], pts_conf, curve_conf_hi, curve_conf_lo,
              "P(correct)", "accuracy by confidence",
              col_hi=C_HICONF, col_lo=C_LOCONF,
              lab_hi="high confidence", lab_lo="low confidence")

        # Both of these panels' curves rise from lower-left to upper-left
        # (P(correct) saturates near 1.0 by the time |orientation| is a few
        # degrees), so an "upper left" legend sits on top of the curves and
        # the earliest data points -- moved to "lower right", which stays
        # empty in both panels since accuracy is already near ceiling there.
        axes[row, 0].legend(frameon=False, loc="lower right", fontsize=FS_LEG)
        axes[row, 3].legend(frameon=False, loc="lower right", fontsize=FS_LEG)

    for ax, s in zip(axes.ravel(), "ABCDEFGH"):
        ax.text(-0.15, 1.08, s, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", va="bottom", ha="left")
    if suptitle:
        fig.suptitle(f"Model: {model}", fontsize=FS_PANEL, fontweight="bold", y=1.01)
    fig.tight_layout()
    # Open up more breathing room between panels than tight_layout's own
    # (fairly tight) default -- tight_layout already set sensible outer
    # margins above; subplots_adjust only widens the internal gaps.
    fig.subplots_adjust(wspace=0.42, hspace=0.55)
    # Per-panel "Ziggy: .../"Friedrich: ..." titles were repetitive (the
    # animal doesn't change across a row's 4 panels) and crowded the title
    # text against the y-axis label of the panel to its right; each row is
    # now labelled once, to the left, the same way Figs. S1/S2 label their
    # model rows.
    for row, animal in enumerate(("Ziggy", "Friedrich")):
        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.065, (pos.y0 + pos.y1) / 2, animal, rotation=90,
                  ha="center", va="center", fontsize=FS_PANEL, fontweight="bold")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=160, bbox_inches="tight")
    print(f"Saved {out_prefix}.pdf / .png")


def safe_name(model):
    """Filesystem/shell-safe filename stem for a model name that may contain
    characters like '|' or '(' (e.g. "Bayesian(|e|)", "FullBayes(e)",
    "Bayesian(crit)", "FullBayes(crit)")."""
    return (model.replace("(|e|)", "_abs_e").replace("(e)", "_e").replace("(crit)", "_crit")
                 .replace("|", "").replace("(", "").replace(")", "").replace(" ", "_")
                 .replace("/", "_"))


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv"
    d = M.load(csv_path)
    fits = {animal: M.fit_animal(d[d.animal == animal]) for animal in ("Ziggy", "Friedrich")}
    _, bin_edges = pd.qcut(d["absori"], 8, duplicates="drop", retbins=True)
    for model in M.ALL_MODELS:
        make_model_figure(fits, d, model, f"figures/modelfit_{safe_name(model)}", bin_edges)
