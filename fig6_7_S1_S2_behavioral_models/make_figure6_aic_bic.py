#!/usr/bin/env python3
"""
Figure 6: AIC/BIC model comparison bar chart (Results,
sec:behavioral-model-comparison; Methods, sec:behavfit-methods).

Two panels, A (Ziggy) and B (Friedrich), each a horizontal bar chart of
Delta-AIC and Delta-BIC (relative to the best-fitting model in that
animal) for all seven confidence models fit directly to real behavior,
sorted by BIC with the best (lowest) model at the top -- matching the
manuscript caption exactly ("models sorted by BIC, best (lowest) at
top"). The bare zero-free-parameter ideal-observer benchmarks that used
to sit alongside these (Bayesian(|e|), FullBayes(e) with no free
criterion at all) are no longer part of the reported model set -- their
one-free-parameter "+criterion" siblings now simply take over the plain
names "Bayesian(|e|)"/"FullBayes(e)" (DISPLAY_NAME below), since a
confidence criterion/bias term is standard practice rather than an
ablation-only add-on.

Reads from model_comparison_aic_bic_corrected.csv (produced by
recompute_aic_bic.py, which re-derives the pure, unpenalized nll for
each fitted model from fit_free_param_models.py's fits -- see that
script's docstring) -- no refitting needed here. The two obsolete
zero-parameter rows (model == "Bayesian(|e|)" or "FullBayes(e)" in that
file, i.e. the pre-rename internal keys) are dropped via OMIT, and the
remaining "Bayesian(crit)"/"FullBayes(crit)" rows are relabeled via
DISPLAY_NAME purely for display -- the underlying AIC/BIC numbers for
every surviving model are unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402

# v2: font sizes and bar colours now come from the shared pub_style
# module -- see shared/pub_style.py. This script previously used
# matplotlib's own unthemed default blue/red (the only figure in the
# paper that hadn't been given its own deliberate palette); it now uses
# pub_style's CMP_A/CMP_B, the generic two-series comparison colour.
# FS_MODEL (the y-axis model-name labels) isn't a pub_style field; it
# maps to PS.FS_TICK since these are tick labels, not a distinct role.
FS_LBL, FS_TICK, FS_TTL, FS_PANEL, FS_MODEL = (
    PS.FS_LBL, PS.FS_TICK, PS.FS_TTL, PS.FS_PANEL, PS.FS_TICK)
C_AIC, C_BIC = PS.CMP_A, PS.CMP_B
# The bare zero-free-parameter ideal-observer rows (pre-rename internal
# keys in model_comparison_aic_bic_corrected.csv), dropped entirely.
OMIT = ("Bayesian(|e|)", "FullBayes(e)")
# Display-only relabeling of the surviving one-free-parameter rows; see
# module docstring.
DISPLAY_NAME = {
    "Bayesian(crit)": "Bayesian(|e|)",
    "FullBayes(crit)": "FullBayes(e)",
}

PS.apply_rcparams()


def panel(ax, rows, title):
    """rows: list of dicts with 'model', 'aic', 'bic' for one animal
    (already filtered to the eight plotted models)."""
    rows = sorted(rows, key=lambda r: r["bic"])  # best (lowest BIC) first
    best_aic = min(r["aic"] for r in rows)
    best_bic = min(r["bic"] for r in rows)
    # Reverse so the best model ends up at the TOP of the horizontal chart
    # (barh plots the first entry at y=0, i.e. the bottom).
    rows = rows[::-1]
    labels = [DISPLAY_NAME.get(r["model"], r["model"]) for r in rows]
    daic = [r["aic"] - best_aic for r in rows]
    dbic = [r["bic"] - best_bic for r in rows]
    y = np.arange(len(rows))
    h = 0.34
    ax.barh(y + h / 2, daic, height=h, color=C_AIC, label=r"$\Delta$AIC",
            zorder=3, edgecolor="white", linewidth=0.4)
    ax.barh(y - h / 2, dbic, height=h, color=C_BIC, label=r"$\Delta$BIC",
            zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_MODEL)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"$\Delta$ (relative to best-fitting model)", fontsize=FS_LBL)
    ax.set_title(title, fontsize=FS_TTL, pad=8)
    ax.axvline(0, color="k", lw=0.9, zorder=2)
    ax.grid(axis="x", color="0.85", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    xmax = max(max(daic), max(dbic), 1.0)
    ax.set_xlim(0, xmax * 1.05)


def make_figure(csv_path="model_comparison_aic_bic_corrected.csv",
                out_prefix="figures/figure6_aic_bic"):
    df = pd.read_csv(csv_path)
    # model_comparison_aic_bic_corrected.csv reports the pure (unpenalized)
    # nll-derived values as aic_pure/bic_pure; normalize to aic/bic here so
    # panel() doesn't need to know about that distinction.
    if "aic_pure" in df.columns:
        df = df.rename(columns={"aic_pure": "aic", "bic_pure": "bic"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, animal, letter in zip(axes, ("Ziggy", "Friedrich"), "AB"):
        sub = df[(df.animal == animal) & (~df.model.isin(OMIT))]
        rows = sub.to_dict("records")
        panel(ax, rows, animal)
        ax.text(-0.32, 1.06, letter, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", va="bottom", ha="left")
    axes[0].legend(frameon=False, loc="upper right", fontsize=PS.FS_LEG, handlelength=1.4)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=160, bbox_inches="tight")
    print(f"Saved {out_prefix}.pdf / .png")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "model_comparison_aic_bic_corrected.csv"
    make_figure(csv_path)
