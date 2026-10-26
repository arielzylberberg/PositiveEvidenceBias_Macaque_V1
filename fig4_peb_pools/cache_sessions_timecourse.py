"""
cache_sessions_timecourse.py
=============================

Time-resolved counterpart of cache_sessions.py: instead of a single
[WIN_START, WIN_END] spike count, extracts spike counts binned in time
(relative to stimulus onset) for every (trial, neuron), and caches the
result to data/session_cache_tc/session_%02d.npz.

This is what makes the time-course panels of Figure 4 (panels C, D)
possible (chosen-/unchosen-pref activity as a function of time, split by
confidence and by accuracy): the existing data/session_cache/*.npz only
stores one number per (trial, neuron) -- the 50-500 ms window count --
which is enough for panel B's bar plot but not for a PSTH.

v2 note: this is a straight port of the original (pre-v4-package) script
of the same name, which read masterNeuralStruct_FULL.mat + neuron_pref.mat
directly via h5py. That version never made it into any code_for_sharing
package, which is why figure1_timecourse.npz previously shipped only as a
precomputed, unregenerable artifact (see data/DATA_DICTIONARY.md). This
version reads the same per-session binned counts from the compact
data/session_NN.npz files instead (via shared/compact_data.py), so no
raw .mat access is needed and it reproduces the shipped
figure1_timecourse.npz bit-for-bit when pointed at the shipped data/.

Default bins: 20 ms, from -100 ms (pre-stimulus baseline) to +600 ms
post-stimulus -- matching data/session_NN.npz's own `timecourse_counts` /
`bin_edges` fields exactly (same convention used elsewhere in the original
MATLAB pipeline: analysis_az/neural_tuning/run_neural_tuning.m's
BIN_SEC=20 ms, analysis_az/PSTHs_sel_nsel/run_PSTH_by_choice_pref.m's
bin_size_sec=20 ms).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import compact_data as CD  # noqa: E402


def cache_sessions_timecourse(data_dir: str, session_indices, cache_dir: str, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    for i in session_indices:
        sd = CD.load_session_timecourse(i, data_dir)
        out = os.path.join(cache_dir, f"session_{i:02d}.npz")
        np.savez(
            out,
            counts_tc=sd["counts_tc"],
            bin_edges=sd["bin_edges"],
            choice=sd["choice"],
            confidence=sd["confidence"],
            correct=sd["correct"],
            right_pref=sd["right_pref"],
            left_pref=sd["left_pref"],
        )
        if verbose:
            print(f"session {i:2d}: cached ({sd['n_trials']} trials x {sd['n_neurons']} neurons "
                  f"x {sd['counts_tc'].shape[-1]} bins)")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else CD.DEFAULT_DATA_DIR
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "session_cache_tc")
    idx = CD.session_indices(data_dir)
    print(f"Caching {len(idx)} sessions from {data_dir} -> {cache_dir}")
    cache_sessions_timecourse(data_dir, idx, cache_dir)
