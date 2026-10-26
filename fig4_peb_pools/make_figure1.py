"""
make_figure1.py
=================

Publication version of the reviewer's Figure 1.

Top panel: a single grouped bar plot (Chosen-pref / Unchosen-pref x
Confidence-split / Accuracy-split) with the two cross-split comparisons
requested in reviewer comment #1 -- chosen-pref (confidence vs. accuracy)
and unchosen-pref (confidence vs. accuracy) -- drawn as significance
brackets directly above the relevant bars, rather than reported as text
below the figure. (The chosen-vs-unchosen comparison within a split is
not of interest and is not annotated.)

If data/figure1_timecourse.npz exists (built by
compute_figure1_timecourse_cached.py), two additional PSTH panels are
appended below the bar panel, in the SAME output file: the time-resolved
analogue of the bar panel, using the same sessions. Each PSTH panel plots
two DIFFERENCE traces -- chosen-pref (High minus Low confidence, or
Correct minus Incorrect) and unchosen-pref (same) -- mean +/- SEM across
sessions, with FDR-corrected (BH, q=0.05) per-bin significance markers
from a one-sample Wilcoxon test of each difference trace against zero.

If figure1_timecourse.npz is absent (e.g. before
cache_sessions_timecourse.py / compute_figure1_timecourse_cached.py have
been run), the figure falls back to the bar panel alone.

All statistics (cross-split bracket p-values, per-bin significance) are
computed here and shown graphically; no statistics text is placed on the
figure itself -- that belongs in the manuscript's figure caption.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_summary import sig_str

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402
PS.apply_rcparams()

HERE = os.path.dirname(os.path.abspath(__file__))

COLOR_CHOSEN = PS.CHOSEN      # matches Fig. 4 chosen-pool color
COLOR_UNCHOSEN = PS.UNCHOSEN  # matches Fig. 4 unchosen-pool color

# x-positions of the four bars: grouped by pool (chosen / unchosen),
# paired within each group by split (confidence / accuracy).
BAR_WIDTH = 0.62
X_CHOSEN_CONF, X_CHOSEN_ACC = 0.0, 0.68
X_UNCHOSEN_CONF, X_UNCHOSEN_ACC = 1.9, 2.58
GROUP_CENTERS = {"Chosen-pref": (X_CHOSEN_CONF + X_CHOSEN_ACC) / 2,
                 "Unchosen-pref": (X_UNCHOSEN_CONF + X_UNCHOSEN_ACC) / 2}

PSTH_SPINE_OFFSET = 6  # pt -- detach the kept left/bottom spines from the
# corner they'd otherwise share (requested for Fig. 4's PSTH panels C/D
# only, not a pub_style-wide change).


def _sig_bracket(ax, x1, x2, y, tick, text):
    """Draw a Tukey-style significance bracket between x1 and x2 with its
    foot at y, and the label centered just above it."""
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color="0.2", linewidth=1.0, clip_on=False, zorder=4)
    ax.text((x1 + x2) / 2, y + tick, text, ha="center", va="bottom",
            fontsize=PS.FS_LEG, clip_on=False)


def _grouped_bar_panel(ax, df):
    """Single panel, four bars grouped by pool (chosen/unchosen) and
    paired by split (confidence/accuracy), with jittered session dots,
    SEM error bars, and significance brackets for the two cross-split
    comparisons (chosen conf vs. acc; unchosen conf vs. acc).

    Split type (confidence vs. accuracy) is distinguished by fill style
    -- solid for confidence, white/open (colored outline only) for
    accuracy -- rather than by hatching, which reads as busier at this
    size."""
    n = len(df)
    series = {
        "chosen_conf": df.chosen_conf_diff.to_numpy(float),
        "chosen_acc": df.chosen_acc_diff.to_numpy(float),
        "unchosen_conf": df.unchosen_conf_diff.to_numpy(float),
        "unchosen_acc": df.unchosen_acc_diff.to_numpy(float),
    }
    xpos = {
        "chosen_conf": X_CHOSEN_CONF, "chosen_acc": X_CHOSEN_ACC,
        "unchosen_conf": X_UNCHOSEN_CONF, "unchosen_acc": X_UNCHOSEN_ACC,
    }
    color = {
        "chosen_conf": COLOR_CHOSEN, "chosen_acc": COLOR_CHOSEN,
        "unchosen_conf": COLOR_UNCHOSEN, "unchosen_acc": COLOR_UNCHOSEN,
    }
    is_acc = {"chosen_conf": False, "chosen_acc": True,
              "unchosen_conf": False, "unchosen_acc": True}

    rng = np.random.default_rng(0)
    for key, vals in series.items():
        x = xpos[key]
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)
        c = color[key]
        if is_acc[key]:
            ax.bar(x, m, width=BAR_WIDTH, facecolor="white", edgecolor=c,
                   linewidth=1.8, zorder=1)
        else:
            ax.bar(x, m, width=BAR_WIDTH, facecolor=c, edgecolor=c,
                   linewidth=1.8, alpha=0.85, zorder=1)
        ax.errorbar(x, m, yerr=sem, fmt="none", ecolor="0.15",
                    elinewidth=1.3, capsize=4, zorder=3)
        jitter = rng.uniform(-0.14, 0.14, size=n)
        ax.scatter(x + jitter, vals, s=14, facecolor=c, edgecolor="none",
                   alpha=0.45, zorder=2)

    ax.axhline(0, color="0.15", linewidth=0.8, zorder=0)

    # Cross-split significance brackets (reviewer comment #1): compare
    # confidence-split vs. accuracy-split, separately within each pool.
    w_ch = stats.wilcoxon(series["chosen_conf"], series["chosen_acc"])
    w_uch = stats.wilcoxon(series["unchosen_conf"], series["unchosen_acc"])

    all_vals = np.concatenate(list(series.values()))
    yr = all_vals.max() - all_vals.min()
    tick = 0.045 * yr

    top_chosen = max(series["chosen_conf"].max(), series["chosen_acc"].max())
    top_unchosen = max(series["unchosen_conf"].max(), series["unchosen_acc"].max())
    _sig_bracket(ax, X_CHOSEN_CONF, X_CHOSEN_ACC, top_chosen + 0.10 * yr,
                 tick, sig_str(w_ch.pvalue))
    _sig_bracket(ax, X_UNCHOSEN_CONF, X_UNCHOSEN_ACC, top_unchosen + 0.10 * yr,
                 tick, sig_str(w_uch.pvalue))

    return w_ch, w_uch


def _sem(x, axis=0):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return x.std(axis=axis, ddof=1) / np.sqrt(n)


def _fdr_sig_bins_onesample(diff, q=0.05):
    """Per-bin one-sample Wilcoxon (across sessions) of `diff`
    (n_sessions, n_bins) against 0, BH-FDR corrected across bins. Returns
    a boolean (n_bins,) mask of significant bins."""
    n_bins = diff.shape[1]
    pvals = np.full(n_bins, np.nan)
    for j in range(n_bins):
        dj = diff[:, j]
        if np.allclose(dj, 0):
            pvals[j] = 1.0
            continue
        try:
            pvals[j] = stats.wilcoxon(dj).pvalue
        except ValueError:
            pvals[j] = 1.0
    sig, _, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return sig, pvals


def _diff_traces(tc, hi_key, lo_key):
    """(n_sessions, n_bins) chosen-pref and unchosen-pref difference
    traces (hi - lo), e.g. High-conf minus Low-conf, or Correct minus
    Incorrect."""
    chosen_diff = tc[f"chosen_{hi_key}"] - tc[f"chosen_{lo_key}"]
    unchosen_diff = tc[f"unchosen_{hi_key}"] - tc[f"unchosen_{lo_key}"]
    return chosen_diff, unchosen_diff


def _psth_panel(ax, tc, hi_key, lo_key, title, ylim, show_legend=False):
    """One PSTH panel: chosen-pref and unchosen-pref DIFFERENCE traces
    (hi - lo, e.g. High-conf minus Low-conf, or Correct minus Incorrect),
    the time-resolved analogue of the bar plot above it. `ylim` is shared
    across panels (computed once in make_figure1) so both PSTH panels use
    the same y-axis."""
    t = tc["bin_centers"] * 1000  # -> ms
    chosen_diff, unchosen_diff = _diff_traces(tc, hi_key, lo_key)

    specs = [
        (chosen_diff, COLOR_CHOSEN, "Chosen-pref"),
        (unchosen_diff, COLOR_UNCHOSEN, "Unchosen-pref"),
    ]
    for arr, color, label in specs:
        m, s = arr.mean(axis=0), _sem(arr)
        ax.plot(t, m, color=color, linewidth=2.2, label=label, zorder=3)
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.15, linewidth=0, zorder=2)

    # Significance rug: a short, thick tick per significant time bin,
    # in its own clean row below the traces (rather than small dotted
    # squares crowding the bottom of the axes).
    sig_chosen, _ = _fdr_sig_bins_onesample(chosen_diff)
    sig_unchosen, _ = _fdr_sig_bins_onesample(unchosen_diff)
    yr = ylim[1] - ylim[0]
    y_chosen_dots = ylim[0] + 0.10 * yr
    y_unchosen_dots = ylim[0] + 0.035 * yr
    rug_h = 0.018 * yr
    if sig_chosen.any():
        ax.vlines(t[sig_chosen], y_chosen_dots - rug_h, y_chosen_dots + rug_h,
                  color=COLOR_CHOSEN, linewidth=2.2, alpha=0.9, clip_on=False, zorder=3)
    if sig_unchosen.any():
        ax.vlines(t[sig_unchosen], y_unchosen_dots - rug_h, y_unchosen_dots + rug_h,
                  color=COLOR_UNCHOSEN, linewidth=2.2, alpha=0.9, clip_on=False, zorder=3)

    ax.set_ylim(ylim)
    ax.axvline(0, color="0.4", linewidth=0.7, zorder=0)
    ax.axhline(0, color="0.4", linewidth=0.6, zorder=0)
    ax.set_title(title, fontsize=PS.FS_TTL, pad=8)
    ax.set_xlabel("Time from stimulus onset (ms)", fontsize=PS.FS_LBL)
    ax.tick_params(labelsize=PS.FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", PSTH_SPINE_OFFSET))
    ax.spines["bottom"].set_position(("outward", PSTH_SPINE_OFFSET))
    if show_legend:
        leg = ax.legend(fontsize=PS.FS_LEG, frameon=True, loc="upper left")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.3")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_alpha(1.0)


def make_figure1(csv_path: str, out_path: str, stats_csv: str | None = None,
                  timecourse_npz: str | None = None):
    df = pd.read_csv(csv_path)

    if timecourse_npz is None:
        timecourse_npz = os.path.join(os.path.dirname(csv_path), "figure1_timecourse.npz")
    have_tc = os.path.exists(timecourse_npz)
    tc = dict(np.load(timecourse_npz)) if have_tc else None

    if have_tc:
        fig = plt.figure(figsize=(9.6, 9.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.32)
        bar_ax = fig.add_subplot(gs[0, :])
        psth_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        panel_letters = ["A", "B", "C"]
    else:
        fig, bar_ax = plt.subplots(figsize=(5, 5))
        psth_axes = None
        panel_letters = ["A"]

    w_ch, w_uch = _grouped_bar_panel(bar_ax, df)

    bar_ax.set_xlim(X_CHOSEN_CONF - 0.55, X_UNCHOSEN_ACC + 0.55)
    bar_ax.set_xticks([X_CHOSEN_CONF, X_CHOSEN_ACC, X_UNCHOSEN_CONF, X_UNCHOSEN_ACC])
    bar_ax.set_xticklabels(["Conf.", "Acc.", "Conf.", "Acc."], fontsize=PS.FS_TICK)
    for group_label, xc in GROUP_CENTERS.items():
        bar_ax.text(xc, -0.10, group_label, transform=bar_ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=PS.FS_LBL)
    bar_ax.tick_params(axis="x", length=0)
    bar_ax.set_ylabel("Mean activity difference\n(z-scored spk)", fontsize=PS.FS_LBL)
    bar_ax.spines[["top", "right"]].set_visible(False)

    ymin, ymax = bar_ax.get_ylim()
    bar_ax.set_ylim(ymin, ymax + 0.07 * (ymax - ymin))  # headroom for bracket labels

    split_legend = [
        Patch(facecolor="0.45", edgecolor="0.45", alpha=0.85, label="Confidence split"),
        Patch(facecolor="white", edgecolor="0.45", linewidth=1.8, label="Accuracy split"),
    ]
    bar_ax.legend(handles=split_legend, fontsize=PS.FS_LEG, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  handlelength=1.5, columnspacing=1.2)

    bar_ax.text(-0.09, 1.16, panel_letters[0], transform=bar_ax.transAxes,
                fontsize=PS.FS_PANEL, fontweight="bold", va="top", ha="left")

    if have_tc:
        # Shared y-limits across both PSTH panels, computed from the
        # mean +/- SEM band of all four difference traces (chosen/
        # unchosen x confidence/accuracy), with extra headroom below for
        # the significance-dot rows.
        def _band_minmax(arr):
            m, s = arr.mean(axis=0), _sem(arr)
            return (m - s).min(), (m + s).max()

        traces = [
            *_diff_traces(tc, "high", "low"),
            *_diff_traces(tc, "correct", "incorrect"),
        ]
        bounds = [_band_minmax(a) for a in traces]
        data_min = min(b[0] for b in bounds)
        data_max = max(b[1] for b in bounds)
        data_yr = data_max - data_min
        psth_ylim = (data_min - 0.20 * data_yr, data_max + 0.10 * data_yr)

        _psth_panel(psth_axes[0], tc, "high", "low",
                    "Confidence split", psth_ylim, show_legend=True)
        _psth_panel(psth_axes[1], tc, "correct", "incorrect",
                    "Accuracy split", psth_ylim, show_legend=False)
        for ax, letter in zip(psth_axes, panel_letters[1:]):
            ax.text(-0.16, 1.14, letter, transform=ax.transAxes,
                    fontsize=PS.FS_PANEL, fontweight="bold", va="top", ha="left")
        psth_axes[0].set_ylabel("Baseline-corrected\nfiring rate (spk/s)", fontsize=PS.FS_LBL)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}" + ("  (bar + time-course panels)" if have_tc else "  (bar panel only)"))
    print(f"  Chosen-pref  conf vs acc: {sig_str(w_ch.pvalue)}  p={w_ch.pvalue:.3g}")
    print(f"  Unchosen-pref  conf vs acc: {sig_str(w_uch.pvalue)}  p={w_uch.pvalue:.3g}")
    return fig


if __name__ == "__main__":
    csv_path = os.path.join(HERE, "data", "figure1_activity.csv")
    out_path = os.path.join(HERE, "figures", "figure1_chosen_unchosen_conf_acc.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = make_figure1(csv_path, out_path)
    out_png = out_path.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
