#!/usr/bin/env python3
"""
extract_session.py -- build compact per-session dataset directly from
masterNeuralStruct_FULL.mat (read in place, never copied). For each
requested session index, extracts:

  - trial_matrix        : raw (31, n_trials_beh) 'trialMatrix', verbatim
  - trial_matrix_all    : raw (31, n_trials_all) 'trialMatrix_all', verbatim,
                          only for sessions that have it
  - trialmatrix_description : (31,) field-name strings, verbatim
  - trial_matrix_index  : dict name -> 1-based row, verbatim
  - expt_date           : string
  - valid_idx           : (n,) 0-based indices into the raw neural trial
                          dimension used for window_counts/timecourse_counts
                          (stim-onset-valid trials, truncated to
                          min(n_trials_neu, n_trials_beh), exactly matching
                          data_io.py's load_session/load_session_timecourse)
  - window_counts       : (n, n_neurons) float32, spikes in [0.05, 0.5] s inclusive
  - timecourse_counts   : (n, n_neurons, 35) int16, 20 ms bins, edges -0.1..0.6 s
  - right_pref, left_pref, p_vals, t_stats, sig_fdr : (n_neurons,) from
                          neuron_pref.json (already-published, single source
                          of truth), truncated to n_neurons = min(n_pref, n_neurons_all)
  - win_start, win_end, bin_edges, n_neurons_all

This is the script that produced every session_NN.npz in this data/
folder, run once directly on the user's machine against the real
masterNeuralStruct_FULL.mat (2.3 GB; never staged/copied off that machine)
and a one-time export of shared/neuron_pref.mat to neuron_pref.json. Kept
here for provenance and in case you ever need to re-extract (e.g. with a
different time window) from the raw file again.

Usage: extract_session.py <session_idx> [<session_idx> ...]
Requires neuron_pref.json next to this script (see shared/neuron_pref.mat
in this package -- export it with:
  python3 -c "
import scipy.io as sio, numpy as np, json
d = sio.loadmat('shared/neuron_pref.mat', squeeze_me=True, struct_as_record=False)
npd, params = d['neuron_pref_data'], d['params']
out = {'params': {'WIN_START': float(params.WIN_START), 'WIN_END': float(params.WIN_END),
                   'ALPHA_SIG': float(params.ALPHA_SIG)}, 'sessions': {}}
for i, s in enumerate(npd):
    out['sessions'][str(i)] = {k: np.asarray(getattr(s, k)).ravel().tolist()
                                for k in ['right_pref','left_pref','p_vals','t_stats','sig_fdr']}
json.dump(out, open('neuron_pref.json','w'))
"
)
Writes session_NN.npz into this same directory.
"""
import sys, os, json, time
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.environ.get(
    "MASTER_NEURAL_STRUCT_FULL",
    os.path.join(HERE, "..", "..", "Boundy-Singer_PNAS2024", "Structures", "masterNeuralStruct_FULL.mat"),
)
OUT_DIR = HERE

WIN_START, WIN_END = 0.05, 0.5
# NOTE: canonical bin edges (np.arange, NOT np.linspace) -- see "Known
# issue, fixed in this version" in README.md. np.linspace(-0.1, 0.6, 36)
# lands on EXACTLY 0.0 and 0.5 (to the bit), while the originally-validated
# analysis_az/comments_followup_conf_acc/cache_sessions_timecourse.py pipeline's
# np.arange(-0.100, 0.601, 0.020) has ~1e-17 floating-point slop at those
# same two edges. That sounds negligible, but a meaningful fraction of
# recorded spikes land at exactly t=0.0 (stimulus onset) or t=0.5s (the
# window-count end), so the two conventions silently put those spikes in
# different 20 ms bins -- enough to measurably bias the baseline-corrected
# PSTH right at the bins that matter most (figure1_timecourse.npz's
# panels C/D). Fixed here to match the canonical/already-validated edges.
BIN_EDGES = np.arange(-0.100, 0.601, 0.020)

with open(os.path.join(HERE, "neuron_pref.json")) as fh:
    PREF = json.load(fh)["sessions"]


def decode_str(ds):
    arr = np.asarray(ds[()]).ravel()
    return "".join(chr(int(c)) for c in arr)


def deref(f, ust, iu, it):
    ref = ust[iu, it]
    arr = f[ref][()]
    return np.asarray(arr).ravel()


def extract_one(f, mns, idx):
    t0 = time.time()
    s = f[mns[idx, 0]]
    beh = s["behavior"]["trialMatrix"]
    tm = beh["trialMatrix"][()]
    tm_all = beh["trialMatrix_all"][()] if "trialMatrix_all" in beh else None
    desc_ds = beh["trialmatrix_description"]
    n_fields = desc_ds.shape[0]
    desc = [decode_str(f[desc_ds[i, 0]]) if desc_ds.dtype == h5py.ref_dtype else str(desc_ds[i]) for i in range(n_fields)]
    tmi_grp = beh["trialMatrix_index"]
    tmi = {name: int(np.asarray(tmi_grp[name][()]).ravel()[0]) for name in tmi_grp.keys()}

    expt_date = decode_str(s["expt"]["date"])

    ds = s["dataStructTask"]
    stim_onsets = ds["stimulus_onsets"][()].ravel()
    valid_stim = ~np.isnan(stim_onsets)
    valid_idx_full = np.where(valid_stim)[0]
    n_trials_neu = len(valid_idx_full)
    n_trials_beh = tm.shape[1]
    n = min(n_trials_neu, n_trials_beh)
    valid_idx = valid_idx_full[:n]

    ust = ds["unit_spike_times"]
    n_neurons_all, n_trials_raw = ust.shape

    pref = PREF[str(idx)]
    right_pref = np.asarray(pref["right_pref"], dtype=bool)
    left_pref = np.asarray(pref["left_pref"], dtype=bool)
    p_vals = np.asarray(pref["p_vals"], dtype=float)
    t_stats = np.asarray(pref["t_stats"], dtype=float)
    sig_fdr = np.asarray(pref["sig_fdr"], dtype=bool)
    n_pref = len(right_pref)
    n_neurons = min(n_pref, n_neurons_all)
    right_pref = right_pref[:n_neurons]
    left_pref = left_pref[:n_neurons]
    p_vals = p_vals[:n_neurons]
    t_stats = t_stats[:n_neurons]
    sig_fdr = sig_fdr[:n_neurons]

    window_counts = np.full((n, n_neurons), np.nan, dtype=np.float32)
    timecourse_counts = np.zeros((n, n_neurons, 35), dtype=np.int16)

    for iu in range(n_neurons):
        for k, it in enumerate(valid_idx):
            st = deref(f, ust, iu, it)
            if st.size == 0:
                window_counts[k, iu] = 0.0
                continue
            window_counts[k, iu] = np.sum((st >= WIN_START) & (st <= WIN_END))
            hist, _ = np.histogram(st, bins=BIN_EDGES)
            timecourse_counts[k, iu, :] = hist

    out_path = os.path.join(OUT_DIR, f"session_{idx:02d}.npz")
    np.savez_compressed(
        out_path,
        idx=idx,
        expt_date=expt_date,
        trial_matrix=tm,
        trial_matrix_all=tm_all if tm_all is not None else np.zeros((0, 0)),
        has_trial_matrix_all=tm_all is not None,
        trialmatrix_description=np.array(desc),
        trial_matrix_index_names=np.array(list(tmi.keys())),
        trial_matrix_index_rows1based=np.array(list(tmi.values()), dtype=int),
        valid_idx=valid_idx.astype(np.int32),
        n=n,
        window_counts=window_counts,
        timecourse_counts=timecourse_counts,
        bin_edges=BIN_EDGES,
        win_start=WIN_START,
        win_end=WIN_END,
        right_pref=right_pref,
        left_pref=left_pref,
        p_vals=p_vals,
        t_stats=t_stats,
        sig_fdr=sig_fdr,
        n_neurons_all=n_neurons_all,
    )
    dt = time.time() - t0
    print(f"session {idx:2d}: OK  n_trials={n:5d}  n_neurons={n_neurons:3d}  "
          f"has_tm_all={tm_all is not None}  [{dt:6.1f}s]  -> {out_path}")


def main():
    idxs = [int(x) for x in sys.argv[1:]]
    f = h5py.File(MAT, "r")
    mns = f["masterNeuralStruct"]
    for idx in idxs:
        extract_one(f, mns, idx)


if __name__ == "__main__":
    main()
