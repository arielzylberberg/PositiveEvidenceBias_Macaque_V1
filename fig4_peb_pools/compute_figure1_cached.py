"""
compute_figure1_cached.py
===========================

Builds data/figure1_activity.csv, the per-session chosen-pref /
unchosen-pref activity-difference table that feeds Figure 4 panel B (the
bar plot; see make_figure1.py / make_figure4_composite.py). Reads
data/session_cache/session_%02d.npz (built by cache_sessions.py in this
folder from the top-level compact data/session_NN.npz files).

MIN_PREF_SIDE=3 (same session-inclusion threshold as Figure 2 / panels
C-D) -- see data/DATA_DICTIONARY.md and the README for why.

Unmodified port of the pre-v4-package script of the same name
(analysis_az/comments_followup_conf_acc/compute_figure1_cached.py); only
the cache format it reads is new (built from the compact data/ folder
instead of a direct masterNeuralStruct_FULL.mat dereference).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_activity import chosen_unchosen_activity

MIN_PREF_SIDE = 3
MIN_TRIALS_TOTAL = 30
MIN_CLASS = 5


class _S:
    pass


def _load_cached(path):
    d = np.load(path)
    s = _S()
    s.counts = d["counts"]
    s.choice = d["choice"]
    s.confidence = d["confidence"]
    s.correct = d["correct"]
    s.right_pref = d["right_pref"].astype(bool)
    s.left_pref = d["left_pref"].astype(bool)
    s.n_trials, s.n_neurons = s.counts.shape
    return s


def run(cache_dir: str, out_csv: str, min_pref_side: int = MIN_PREF_SIDE, verbose=True):
    rows = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "session_*.npz"))):
        i = int(os.path.basename(path).replace("session_", "").replace(".npz", ""))
        sd = _load_cached(path)

        valid = (
            np.isin(sd.choice, [-1, 1])
            & np.isin(sd.confidence, [0, 1])
            & ~np.isnan(sd.correct)
        )
        n_r, n_l = sd.right_pref.sum(), sd.left_pref.sum()

        if n_r < min_pref_side or n_l < min_pref_side:
            if verbose:
                print(f"session {i:2d}: SKIP  (R-pref={n_r} L-pref={n_l})")
            continue
        if valid.sum() < MIN_TRIALS_TOTAL:
            if verbose:
                print(f"session {i:2d}: SKIP  (only {valid.sum()} valid trials)")
            continue

        conf_v = sd.confidence[valid]
        corr_v = sd.correct[valid]
        n_high, n_low = (conf_v == 1).sum(), (conf_v == 0).sum()
        n_corr, n_incorr = (corr_v == 1).sum(), (corr_v == 0).sum()
        if min(n_high, n_low, n_corr, n_incorr) < MIN_CLASS:
            if verbose:
                print(f"session {i:2d}: SKIP  (class counts too small)")
            continue

        x_chosen, x_unchosen, _ = chosen_unchosen_activity(sd, valid)

        def diff(x, g_hi, g_lo):
            a = x[valid][g_hi]
            b = x[valid][g_lo]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            return a.mean() - b.mean()

        hi, lo = conf_v == 1, conf_v == 0
        co, ic = corr_v == 1, corr_v == 0

        rows.append(dict(
            session=i,
            n_trials=int(valid.sum()),
            n_right_pref=int(n_r),
            n_left_pref=int(n_l),
            chosen_conf_diff=diff(x_chosen, hi, lo),
            unchosen_conf_diff=diff(x_unchosen, hi, lo),
            chosen_acc_diff=diff(x_chosen, co, ic),
            unchosen_acc_diff=diff(x_unchosen, co, ic),
        ))
        if verbose:
            print(f"session {i:2d}: OK  n={rows[-1]['n_trials']}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} sessions included (MIN_PREF_SIDE={min_pref_side}) -> {out_csv}")
    return df


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "session_cache")
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "figure1_activity.csv")
    run(CACHE_DIR, OUT)
