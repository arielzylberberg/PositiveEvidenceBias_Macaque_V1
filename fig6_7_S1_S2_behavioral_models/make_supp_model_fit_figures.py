#!/usr/bin/env python3
"""
Figures S1/S2: composite behavioral-fit figures for the remaining seven
fitted confidence models (Methods, sec:supp-model-fits), one row PER MODEL
(not per animal, unlike plot_model_fits.py's individual per-model
figures): each row has 8 columns, the same four panel types as Fig. 7,
first for Ziggy then for Friedrich --

  1. accuracy (all trials), split by contrast
  2. confidence, correct trials only, split by contrast
  3. confidence, error trials only, split by contrast
  4. accuracy (all trials), split by reported confidence

repeated for each animal. This matches the manuscript caption exactly:
"one row per model... each animal contributes four columns... the
accuracy-by-contrast and accuracy-by-confidence columns are identical
across all seven models shown here and in [the other supp. figure]... and
are repeated on every row for ease of comparison."

Reuses plot_model_fits.py's binning/curve/panel-drawing helpers directly
(imported as P) so S1/S2 stay pixel-for-pixel consistent with Fig. 7 and
the individual per-model figures (figures/modelfit_*.pdf) -- only the
figure layout (rows-of-models instead of one-model-per-figure) differs.

  Fig. S1 (figureS1_model_fits_1): Calibrated, Bayesian(|e|)
    [code name Bayesian(crit); see DISPLAY_NAME below], ContrastBlind,
    Entropy(e).
  Fig. S2 (figureS2_model_fits_2): CASANDRE, FullBayes(e)
    [code name FullBayes(crit); see DISPLAY_NAME below]. (The retired
    "PositiveEvidence" model, a nominal-stimulus-only read-out, is no
    longer part of the fitted model set -- see fit_free_param_models.py
    module docstring item (2).)

The bare zero-free-parameter ideal-observer benchmarks that used to sit
alongside these (the literal Bayesian(|e|) and FullBayes(e), with no
free criterion at all) are no longer part of the reported model set: the
one-free-parameter "+criterion" versions now simply take over the plain
names "Bayesian(|e|)" and "FullBayes(e)" (DISPLAY_NAME below), since a
confidence criterion/bias term is standard practice rather than an
ablation-only add-on. The internal fit lookups still use the original
"Bayesian(crit)"/"FullBayes(crit)" keys; only the drawn row label changes.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import fit_free_param_models as M
import plot_model_fits as P

S1_MODELS = ["Calibrated", "Bayesian(crit)", "ContrastBlind", "Entropy(e)"]
S2_MODELS = ["CASANDRE", "FullBayes(crit)"]

# Display-only relabeling: the manuscript no longer includes the bare
# zero-free-parameter "Bayesian(|e|)"/"FullBayes(e)" ideal-observer
# benchmarks in the model set at all (they added little beyond showing
# that a single free criterion earns its keep, which is no longer part of
# the reported comparison); "Bayesian(crit)"/"FullBayes(crit)" -- the
# one-free-parameter versions -- now simply take over the plain names
# "Bayesian(|e|)"/"FullBayes(e)". This dict only changes what is drawn as
# the row label; the underlying fit lookups above still use the original
# internal keys, so no fits are re-run or reinterpreted.
DISPLAY_NAME = {
    "Bayesian(crit)": "Bayesian(|e|)",
    "FullBayes(crit)": "FullBayes(e)",
}

ROW_H = 3.5    # inches per model row
COL_W = 3.0    # inches per panel column (8 columns per row -> figure width = 8*COL_W)
FS_CAT_TTL = 10.5  # per-panel category title (row 0 only; animal name is a separate header)

# Shared per-column category titles -- shown once, on the top row only, since
# they are identical for every model row (see module docstring).
CATEGORY_TITLES = ["accuracy (all trials)", "confidence, correct only",
                   "confidence, error only", "accuracy by confidence"]


def draw_animal_block(axes4, fit, model, d_animal, bin_edges, animal, show_legend, show_titles):
    """Fill 4 axes for one animal x model, mirroring
    plot_model_fits.make_model_figure's per-animal column logic exactly
    (titles are shown only for the top model row -- see CATEGORY_TITLES)."""
    titles = CATEGORY_TITLES if show_titles else ("", "", "", "")

    # secondary_encoding=True on the three contrast-split panels: see
    # plot_model_fits.py's module docstring -- CONTRAST_HI/LO and
    # CONF_HIGH/LOW both appear in this figure, and their CVD separation
    # is only borderline (Delta-E 6.1), so the contrast split also gets a
    # dashed-line/open-marker secondary encoding on top of hue.
    pts_acc = P.binned_points(d_animal, "correct", bin_edges, splitcol="hi")
    curve_acc_hi = M.predict_accuracy_curve(fit, 1)
    curve_acc_lo = M.predict_accuracy_curve(fit, 0)
    P.panel(axes4[0], pts_acc, curve_acc_hi, curve_acc_lo, "P(correct)", titles[0],
            secondary_encoding=True)

    d_corr = d_animal[d_animal.correct == 1]
    pts_c = P.binned_points(d_corr, "confidence", bin_edges, splitcol="hi")
    curve_c_hi = M.predict_confidence_curve(fit, model, 1, 1)
    curve_c_lo = M.predict_confidence_curve(fit, model, 0, 1)
    P.panel(axes4[1], pts_c, curve_c_hi, curve_c_lo, "P(high confidence)", titles[1],
            secondary_encoding=True)

    d_err = d_animal[d_animal.correct == 0]
    pts_e = P.binned_points(d_err, "confidence", bin_edges, splitcol="hi")
    curve_e_hi = M.predict_confidence_curve(fit, model, 1, 0)
    curve_e_lo = M.predict_confidence_curve(fit, model, 0, 0)
    curve_e_hi = P.truncate_curve_to_data(curve_e_hi, pts_e)
    curve_e_lo = P.truncate_curve_to_data(curve_e_lo, pts_e)
    P.panel(axes4[2], pts_e, curve_e_hi, curve_e_lo, "P(high confidence)", titles[2],
            secondary_encoding=True)

    ylo, yhi = P.combined_ylim(pts_c, pts_e, curve_c_hi, curve_c_lo, curve_e_hi, curve_e_lo)
    axes4[1].set_ylim(ylo, yhi)
    axes4[2].set_ylim(ylo, yhi)

    pts_conf = P.binned_points(d_animal, "correct", bin_edges, splitcol="confidence")
    s_all = d_animal["absori"].to_numpy()
    conf_all = d_animal["confidence"].to_numpy()
    corr_all = d_animal["correct"].to_numpy()
    a_hi = P.fit_noint_probit(s_all[conf_all == 1], corr_all[conf_all == 1])
    a_lo = P.fit_noint_probit(s_all[conf_all == 0], corr_all[conf_all == 0])
    gs = fit["grid_s"]
    curve_conf_hi = (gs, norm.cdf(a_hi * gs))
    curve_conf_lo = (gs, norm.cdf(a_lo * gs))
    P.panel(axes4[3], pts_conf, curve_conf_hi, curve_conf_lo, "P(correct)", titles[3],
            col_hi=P.C_HICONF, col_lo=P.C_LOCONF,
            lab_hi="high confidence", lab_lo="low confidence")

    for ax in axes4:
        ax.title.set_fontsize(FS_CAT_TTL)
    if show_legend:
        legend_kw = dict(loc="lower right", fontsize=P.FS_LEG - 1, frameon=True,
                          edgecolor="0.6", facecolor="white", framealpha=0.95)
        axes4[0].legend(**legend_kw)
        axes4[3].legend(**legend_kw)


def make_supp_figure(fits, data, models, out_prefix, bin_edges, title=None):
    """title is accepted but no longer drawn on the figure itself -- it was
    redundant with the manuscript's own \\mycaption (e.g. "Figure S1:
    behavioral fits for the remaining confidence models (1 of 2)"), kept
    as a parameter only so callers don't need updating."""
    n = len(models)
    animals = ("Ziggy", "Friedrich")
    # 9-column gridspec: 4 Ziggy columns, 1 narrow unused spacer column, 4
    # Friedrich columns. GridSpec only supports one uniform wspace across
    # all columns, so a real (not just visually-implied) extra gap between
    # the two animals' blocks needs its own column -- left empty -- rather
    # than a wider wspace applied everywhere (which would also push the
    # four panels within one animal's own block further apart).
    fig = plt.figure(figsize=(COL_W * 8.7, ROW_H * n))
    width_ratios = [1, 1, 1, 1, 0.55, 1, 1, 1, 1]
    gs = fig.add_gridspec(n, 9, width_ratios=width_ratios, wspace=0.38, hspace=0.55)
    axes = np.empty((n, 8), dtype=object)
    for row in range(n):
        for col in range(8):
            gcol = col if col < 4 else col + 1  # skip the spacer at index 4
            axes[row, col] = fig.add_subplot(gs[row, gcol])

    for row, model in enumerate(models):
        for block, animal in enumerate(animals):
            fit = fits[animal]
            d_animal = data[data.animal == animal]
            axes4 = axes[row, block * 4:(block + 1) * 4]
            draw_animal_block(axes4, fit, model, d_animal, bin_edges, animal,
                               show_legend=(row == 0), show_titles=(row == 0))
    fig.tight_layout()
    # Row labels (model names) and animal-block headers, placed once final
    # axes positions are known (must come after tight_layout).
    for row, model in enumerate(models):
        pos = axes[row, 0].get_position()
        label = DISPLAY_NAME.get(model, model)
        fig.text(pos.x0 - 0.05, (pos.y0 + pos.y1) / 2, label, rotation=90,
                  ha="center", va="center", fontsize=P.FS_TTL, fontweight="bold")
    for block, animal in enumerate(animals):
        pos_l = axes[0, block * 4].get_position()
        pos_r = axes[0, block * 4 + 3].get_position()
        fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.05 / n, animal,
                  ha="center", va="bottom", fontsize=P.FS_PANEL, fontweight="bold")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=160, bbox_inches="tight")
    print(f"Saved {out_prefix}.pdf / .png")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv"
    d = M.load(csv_path)
    fits = {animal: M.fit_animal(d[d.animal == animal]) for animal in ("Ziggy", "Friedrich")}
    _, bin_edges = pd.qcut(d["absori"], 8, duplicates="drop", retbins=True)

    make_supp_figure(fits, d, S1_MODELS, "figures/figureS1_model_fits_1", bin_edges)
    make_supp_figure(fits, d, S2_MODELS, "figures/figureS2_model_fits_2", bin_edges)
