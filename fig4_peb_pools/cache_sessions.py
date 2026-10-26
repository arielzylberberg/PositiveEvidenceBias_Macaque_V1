"""
cache_sessions.py
==================

v2 (this package): builds data/session_cache/session_%02d.npz (the window
spike-count format compute_figure1_cached.py reads, feeding Figure 4
panel B) from the top-level, already-compact ../../data/session_NN.npz
files instead of dereferencing masterNeuralStruct_FULL.mat directly. The
slow part (one HDF5 object-reference dereference per neuron x trial) was
already done once, for every session, when ../../data/session_NN.npz was
built (see ../../data/extract_session.py) -- this script now just reads
back the window spike counts already stored there, so it runs in well
under a second for all 29 sessions instead of the many minutes the
original, FULL-dereferencing version needed.

Identical in content to fig2_gain_maps/cache_sessions.py (same compact
data/ -> session_cache/ reshaping is needed in both places).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import compact_data as CD  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def cache_sessions(session_indices=None, cache_dir=None, data_dir=None, verbose=True):
    data_dir = data_dir or CD.DEFAULT_DATA_DIR
    cache_dir = cache_dir or os.path.join(HERE, "data", "session_cache")
    session_indices = session_indices if session_indices is not None else CD.session_indices(data_dir)

    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    for i in session_indices:
        sd = CD.load_session(i, data_dir)
        out = os.path.join(cache_dir, f"session_{i:02d}.npz")
        np.savez(
            out,
            counts=sd["counts"],
            choice=sd["choice"],
            confidence=sd["confidence"],
            correct=sd["correct"],
            right_pref=sd["right_pref"],
            left_pref=sd["left_pref"],
            win_start=sd["win_start"],
            win_end=sd["win_end"],
        )
        if verbose:
            print(f"session {i:2d}: cached ({sd['n_trials']} trials x {sd['n_neurons']} neurons)"
                  f"  [{time.time()-t0:.1f}s elapsed]")


if __name__ == "__main__":
    idx = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
    cache_sessions(idx)
