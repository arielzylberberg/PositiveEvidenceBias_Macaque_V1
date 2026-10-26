"""
compute_activity.py
====================

Core computations shared by Figure 4 panel B (chosen/unchosen activity,
High-Low confidence and Correct-Incorrect bar plot, via
compute_figure1_cached.py) and panels C/D (via
compute_figure1_timecourse_cached.py). Both start from the same building
block: per-neuron z-scored spike counts in [WIN_START, WIN_END], averaged
within the "chosen-pref" pool (neurons preferring the choice made) and
"unchosen-pref" pool (neurons preferring the choice NOT made), per trial.

"correct" is defined as choice == CT_X (trialMatrix column 10), matching
the convention established in data_io.py / shared/compact_data.py.

Unmodified port of the pre-v4-package script of the same name
(analysis_az/comments_followup_conf_acc/compute_activity.py).
"""
from __future__ import annotations

import numpy as np


def zscore_columns(counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score each column (neuron) of `counts` using only `mask` rows,
    matching the MATLAB `mu/sg` per-neuron loop (min 10 valid trials,
    skip if std==0)."""
    z = np.full_like(counts, np.nan, dtype=float)
    sub = counts[mask, :]
    for iu in range(counts.shape[1]):
        col = sub[:, iu]
        ok = ~np.isnan(col)
        if ok.sum() < 10:
            continue
        mu, sg = col[ok].mean(), col[ok].std(ddof=0)
        if sg == 0:
            continue
        z[:, iu] = (counts[:, iu] - mu) / sg
    return z


def nanmean_cols(z: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Row-wise mean over the columns selected by boolean vector `sel`.
    Returns NaN for every row if `sel` selects zero columns (matches
    MATLAB nanmean(cz_v(:, empty), 2) -> NaN)."""
    if sel.sum() == 0:
        return np.full(z.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(z[:, sel], axis=1)


def chosen_unchosen_activity(sd, valid_mask: np.ndarray):
    """Per-trial chosen-pref / unchosen-pref / non-selective z-scored
    activity, z-scored using `valid_mask` trials, for ALL trials in the
    session (NaN where undefined).

    Returns (x_chosen, x_unchosen, x_nonsel) each shape (n_trials,).
    """
    z = zscore_columns(sd.counts, valid_mask)

    non_sel = ~sd.right_pref & ~sd.left_pref

    x_right = nanmean_cols(z, sd.right_pref)
    x_left = nanmean_cols(z, sd.left_pref)
    x_ns = nanmean_cols(z, non_sel)

    choice = sd.choice
    right_v = choice == 1
    left_v = choice == -1

    x_chosen = np.full(sd.n_trials, np.nan)
    x_unchosen = np.full(sd.n_trials, np.nan)
    x_chosen[right_v] = x_right[right_v]
    x_chosen[left_v] = x_left[left_v]
    x_unchosen[right_v] = x_left[right_v]
    x_unchosen[left_v] = x_right[left_v]

    return x_chosen, x_unchosen, x_ns
