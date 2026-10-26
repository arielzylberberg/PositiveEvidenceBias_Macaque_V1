"""
data_io.py
==========

Python data-loading layer for masterNeuralStruct_FULL.mat (HDF5 / MATLAB
v7.3), re-implementing the exact conventions already established by the
MATLAB pipeline in analysis_az/ (compute_neuron_pref.m,
run_confidence_regression.m, run_confidence_regression_accuracy.m,
compute_ab_marginal.m):

  * trialMatrix / trialMatrix_index : behavioral trial matrix + a
    name->row lookup (1-based in MATLAB; converted to 0-based here).
  * choice   = SIDE column        (-1 = Left/CCW, +1 = Right/CW)
  * confidence = CONFIDENCE column (0 = Low, 1 = High)
  * "correct"  = choice == CT_X column (trialMatrix column 10 in MATLAB
    1-based indexing -- the X-location of the correct-choice target),
    matching run_confidence_regression_accuracy.m / run_PSTH_by_choice_pref.m
  * spike counts: sum of unit_spike_times (seconds, relative to stimulus
    onset) inside [WIN_START, WIN_END] (loaded from
    analysis_az/neuron_pref/neuron_pref.mat's saved params, default
    0.050-0.500 s)
  * neuron_pref.mat's right_pref / left_pref boolean vectors are the
    single source of truth for which neurons are "right-preferring" /
    "left-preferring" -- loaded as-is (NOT recomputed), so results are
    numerically identical to the existing MATLAB pipeline.

Only trials with a valid stimulus onset (matching MATLAB's
`I = ~isnan(ds.stimulus_onsets)` filter) are kept, mirroring the
trial-count reconciliation already done in the MATLAB scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import h5py
import scipy.io as sio

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
# REPO_ROOT here is .../outputs/conf_acc_replication/../.. -> outputs/  (wrong
# when run from the scratch folder). The real repo root is passed in
# explicitly by run_all.py via `set_repo_root`, so this default is only a
# fallback for ad-hoc testing.


@dataclass
class SessionData:
    idx: int
    n_neurons: int
    n_trials: int
    choice: np.ndarray          # (-1, +1), NaN not allowed (only valid trials kept upstream? no: full array)
    confidence: np.ndarray      # 0/1 or NaN
    correct: np.ndarray         # 0/1 or NaN (float, NaN where CT_X==0 i.e. undefined)
    counts: np.ndarray          # (n_trials, n_neurons) raw spike counts in window
    right_pref: np.ndarray      # bool (n_neurons,)
    left_pref: np.ndarray       # bool (n_neurons,)


@dataclass
class SessionTimecourse:
    """Same trial/neuron bookkeeping as SessionData, but with spike counts
    binned in time (relative to stimulus onset) instead of summed in a
    single [WIN_START, WIN_END] window. Produced by
    `load_session_timecourse` / cached by cache_sessions_timecourse.py."""
    idx: int
    n_neurons: int
    n_trials: int
    n_bins: int
    bin_edges: np.ndarray       # (n_bins + 1,) seconds relative to stim onset
    choice: np.ndarray          # (-1, +1) or NaN
    confidence: np.ndarray      # 0/1 or NaN
    correct: np.ndarray         # 0/1 or NaN
    counts_tc: np.ndarray       # (n_trials, n_neurons, n_bins) raw spike counts per bin
    right_pref: np.ndarray      # bool (n_neurons,)
    left_pref: np.ndarray       # bool (n_neurons,)


def load_neuron_pref(pref_path: str):
    """Load analysis_az/neuron_pref/neuron_pref.mat (small, classic .mat)."""
    d = sio.loadmat(pref_path, squeeze_me=True, struct_as_record=False)
    npd = d["neuron_pref_data"]
    params = d["params"]
    win_start = float(params.WIN_START)
    win_end = float(params.WIN_END)
    prefs = []
    for s in npd:
        right_pref = np.asarray(s.right_pref).astype(bool).ravel()
        left_pref = np.asarray(s.left_pref).astype(bool).ravel()
        prefs.append((right_pref, left_pref))
    return prefs, win_start, win_end


def _deref_spike_times(f, ust_dataset, iu, it):
    ref = ust_dataset[iu, it]
    arr = f[ref][()]
    return arr.ravel()


def load_session(f: h5py.File, mns, idataset: int, win_start: float, win_end: float,
                  right_pref: np.ndarray, left_pref: np.ndarray) -> "SessionData | None":
    """Load + window-count one session (0-based idataset)."""
    s = f[mns[idataset, 0]]

    tmg = s["behavior"]["trialMatrix"]
    tmi_grp = tmg["trialMatrix_index"]
    tm = tmg["trialMatrix"][()]  # (n_fields, n_trials_beh)  -- MATLAB tm(:,field) == tm[field-1, :]

    def col(name):
        row1 = int(tmi_grp[name][()].ravel()[0])  # 1-based MATLAB row index
        return tm[row1 - 1, :]

    choice_raw = col("SIDE")
    conf_raw = col("CONFIDENCE")
    ctx_raw = col("CT_X")  # "column 10" in MATLAB 1-based == CT_X

    ds = s["dataStructTask"]
    stim_onsets = ds["stimulus_onsets"][()].ravel()
    valid_stim = ~np.isnan(stim_onsets)
    n_valid_stim = int(valid_stim.sum())

    ust = ds["unit_spike_times"]
    n_neurons_all, n_trials_raw = ust.shape

    # Reconcile trial counts exactly like the MATLAB scripts do: spk_times
    # is indexed by the stim-onset validity mask; if that count doesn't
    # match the behavioral trial matrix length, truncate both to the min.
    valid_idx = np.where(valid_stim)[0]
    n_trials_neu = len(valid_idx)
    n_trials_beh = tm.shape[1]
    n = min(n_trials_neu, n_trials_beh)
    valid_idx = valid_idx[:n]
    choice_raw = choice_raw[:n]
    conf_raw = conf_raw[:n]
    ctx_raw = ctx_raw[:n]

    n_pref = len(right_pref)
    n_neurons = min(n_pref, n_neurons_all)
    right_pref = right_pref[:n_neurons]
    left_pref = left_pref[:n_neurons]

    counts = np.full((n, n_neurons), np.nan)
    for iu in range(n_neurons):
        for k, it in enumerate(valid_idx):
            st = _deref_spike_times(f, ust, iu, it)
            if st.size == 0:
                continue
            counts[k, iu] = np.sum((st >= win_start) & (st <= win_end))

    correct = np.full(n, np.nan)
    has_target = ctx_raw != 0
    correct[has_target] = (choice_raw[has_target] == ctx_raw[has_target]).astype(float)

    return SessionData(
        idx=idataset,
        n_neurons=n_neurons,
        n_trials=n,
        choice=choice_raw,
        confidence=conf_raw,
        correct=correct,
        counts=counts,
        right_pref=right_pref,
        left_pref=left_pref,
    )


def load_session_timecourse(f: h5py.File, mns, idataset: int, bin_edges: np.ndarray,
                             right_pref: np.ndarray, left_pref: np.ndarray) -> "SessionTimecourse":
    """Load + time-bin one session (0-based idataset).

    Identical trial/neuron reconciliation logic to `load_session` (same
    stim-onset validity mask, same min(n_trials_neu, n_trials_beh)
    truncation, same right_pref/left_pref slicing) so that a session's
    trial order and neuron order here line up exactly with the existing
    [WIN_START, WIN_END] cache -- only the spike-counting step differs:
    instead of one sum over a fixed window, spikes are histogrammed into
    `bin_edges` (seconds relative to stimulus onset; may start negative
    for a pre-stimulus baseline). This is the same, single expensive
    dereference of `unit_spike_times` per (neuron, trial) as
    `load_session` -- binning is a cheap `np.histogram` on the already
    -fetched spike times, so caching the time course costs about the same
    wall-clock time as caching the single-window counts already cached by
    cache_sessions.py.
    """
    s = f[mns[idataset, 0]]

    tmg = s["behavior"]["trialMatrix"]
    tmi_grp = tmg["trialMatrix_index"]
    tm = tmg["trialMatrix"][()]

    def col(name):
        row1 = int(tmi_grp[name][()].ravel()[0])
        return tm[row1 - 1, :]

    choice_raw = col("SIDE")
    conf_raw = col("CONFIDENCE")
    ctx_raw = col("CT_X")

    ds = s["dataStructTask"]
    stim_onsets = ds["stimulus_onsets"][()].ravel()
    valid_stim = ~np.isnan(stim_onsets)

    ust = ds["unit_spike_times"]
    n_neurons_all, n_trials_raw = ust.shape

    valid_idx = np.where(valid_stim)[0]
    n_trials_neu = len(valid_idx)
    n_trials_beh = tm.shape[1]
    n = min(n_trials_neu, n_trials_beh)
    valid_idx = valid_idx[:n]
    choice_raw = choice_raw[:n]
    conf_raw = conf_raw[:n]
    ctx_raw = ctx_raw[:n]

    n_pref = len(right_pref)
    n_neurons = min(n_pref, n_neurons_all)
    right_pref = right_pref[:n_neurons]
    left_pref = left_pref[:n_neurons]

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1

    counts_tc = np.full((n, n_neurons, n_bins), np.nan)
    for iu in range(n_neurons):
        for k, it in enumerate(valid_idx):
            st = _deref_spike_times(f, ust, iu, it)
            if st.size == 0:
                counts_tc[k, iu, :] = 0.0
                continue
            hist, _ = np.histogram(st, bins=bin_edges)
            counts_tc[k, iu, :] = hist

    correct = np.full(n, np.nan)
    has_target = ctx_raw != 0
    correct[has_target] = (choice_raw[has_target] == ctx_raw[has_target]).astype(float)

    return SessionTimecourse(
        idx=idataset,
        n_neurons=n_neurons,
        n_trials=n,
        n_bins=n_bins,
        bin_edges=bin_edges,
        choice=choice_raw,
        confidence=conf_raw,
        correct=correct,
        counts_tc=counts_tc,
        right_pref=right_pref,
        left_pref=left_pref,
    )
