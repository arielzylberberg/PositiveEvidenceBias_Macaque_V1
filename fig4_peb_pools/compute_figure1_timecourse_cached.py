"""
compute_figure1_timecourse_cached.py
======================================

Time-resolved counterpart of the single-window chosen/unchosen activity
computation: instead of one [50,500] ms activity difference per session,
computes a full baseline-corrected chosen-pref / unchosen-pref PSTH per
session, split by confidence (High/Low) and by accuracy (Correct/
Incorrect). Feeds Figure 4's panels C and D (make_figure4_composite.py)
via make_figure1.py's PSTH panel code.

Pipeline (per session, reading data/session_cache_tc/session_%02d.npz,
built by cache_sessions_timecourse.py):
  1. Session-inclusion criteria (>=3 right-pref AND >=3 left-pref
     neurons, >=30 valid trials, >=5 trials in each of {High,Low} and
     {Correct,Incorrect}) -- the SAME 18 sessions as Figure 4 panel B.
  2. Convert binned raw spike counts to firing rate (spk/s).
  3. Baseline-correct each neuron by subtracting its own mean rate in the
     pre-stimulus bins (bin edges < 0).
  4. Re-derive the chosen-pref / unchosen-pref pools per time bin: for
     right-choice trials, chosen = mean over right-preferring neurons,
     unchosen = mean over left-preferring neurons (and vice-versa for
     left-choice trials).
  5. Average within each session over trials of a given condition
     (High-conf / Low-conf / Correct / Incorrect), separately for the
     chosen-pref and unchosen-pref pools --> 8 per-session PSTHs.

Output: data/figure1_timecourse.npz with one (n_sessions, n_bins) array
per condition, plus bin_centers and the session list.

This is an unmodified port of the pre-v4-package script of the same name
(analysis_az/comments_followup_conf_acc/compute_figure1_timecourse_cached.py)
-- only the session-cache format it reads is unchanged, so it reproduces
the shipped figure1_timecourse.npz bit-for-bit when run against
data/session_cache_tc built from the shipped neuron_pref labels.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_PREF_SIDE = 3
MIN_TRIALS_TOTAL = 30
MIN_CLASS = 5


class _S:
    pass


def _load_cached(path):
    d = np.load(path)
    s = _S()
    s.counts_tc = d["counts_tc"]          # (n_trials, n_neurons, n_bins)
    s.bin_edges = d["bin_edges"]
    s.choice = d["choice"]
    s.confidence = d["confidence"]
    s.correct = d["correct"]
    s.right_pref = d["right_pref"].astype(bool)
    s.left_pref = d["left_pref"].astype(bool)
    s.n_trials, s.n_neurons, s.n_bins = s.counts_tc.shape
    return s


def _baseline_corrected_rate(sd, valid_mask):
    """(n_trials, n_neurons, n_bins) baseline-corrected firing rate.

    Baseline = each neuron's own mean rate (spk/s) in the pre-stimulus
    bins (bin_edges < 0), averaged over `valid_mask` trials.
    """
    bin_width = np.diff(sd.bin_edges)
    rate = sd.counts_tc / bin_width[None, None, :]   # spk/s

    bin_centers = (sd.bin_edges[:-1] + sd.bin_edges[1:]) / 2
    pre_bins = bin_centers < 0
    if not pre_bins.any():
        raise ValueError("bin_edges must include at least one pre-stimulus bin "
                          "(bin center < 0) to define the baseline.")

    baseline = np.nanmean(rate[valid_mask][:, :, pre_bins], axis=(0, 2))  # (n_neurons,)
    return rate - baseline[None, :, None], bin_centers


def _pool_timecourse(sd, rate_bc):
    """(n_trials, n_bins) chosen-pref / unchosen-pref traces, per trial."""
    x_right = np.nanmean(rate_bc[:, sd.right_pref, :], axis=1) if sd.right_pref.any() \
        else np.full((sd.n_trials, sd.n_bins), np.nan)
    x_left = np.nanmean(rate_bc[:, sd.left_pref, :], axis=1) if sd.left_pref.any() \
        else np.full((sd.n_trials, sd.n_bins), np.nan)

    right_v = sd.choice == 1
    left_v = sd.choice == -1

    x_chosen = np.full((sd.n_trials, sd.n_bins), np.nan)
    x_unchosen = np.full((sd.n_trials, sd.n_bins), np.nan)
    x_chosen[right_v] = x_right[right_v]
    x_chosen[left_v] = x_left[left_v]
    x_unchosen[right_v] = x_left[right_v]
    x_unchosen[left_v] = x_right[left_v]
    return x_chosen, x_unchosen


def run(cache_dir: str, out_npz: str, min_pref_side: int = MIN_PREF_SIDE, verbose=True):
    rows = dict(session=[], chosen_high=[], chosen_low=[], unchosen_high=[], unchosen_low=[],
                chosen_correct=[], chosen_incorrect=[], unchosen_correct=[], unchosen_incorrect=[])
    bin_centers = None

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

        rate_bc, bc = _baseline_corrected_rate(sd, valid)
        bin_centers = bc
        x_chosen, x_unchosen = _pool_timecourse(sd, rate_bc)

        hi, lo = conf_v == 1, conf_v == 0
        co, ic = corr_v == 1, corr_v == 0

        def cond_mean(x, sel):
            return np.nanmean(x[valid][sel], axis=0)

        rows["session"].append(i)
        rows["chosen_high"].append(cond_mean(x_chosen, hi))
        rows["chosen_low"].append(cond_mean(x_chosen, lo))
        rows["unchosen_high"].append(cond_mean(x_unchosen, hi))
        rows["unchosen_low"].append(cond_mean(x_unchosen, lo))
        rows["chosen_correct"].append(cond_mean(x_chosen, co))
        rows["chosen_incorrect"].append(cond_mean(x_chosen, ic))
        rows["unchosen_correct"].append(cond_mean(x_unchosen, co))
        rows["unchosen_incorrect"].append(cond_mean(x_unchosen, ic))
        if verbose:
            print(f"session {i:2d}: OK  n={int(valid.sum())}")

    n_sessions = len(rows["session"])
    out = {k: (np.array(v) if k != "session" else np.array(v, dtype=int))
           for k, v in rows.items()}
    out["bin_centers"] = bin_centers

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **out)
    print(f"\n{n_sessions} sessions included (MIN_PREF_SIDE={min_pref_side}) -> {out_npz}")
    return out


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "session_cache_tc")
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "figure1_timecourse.npz")
    run(CACHE_DIR, OUT)
