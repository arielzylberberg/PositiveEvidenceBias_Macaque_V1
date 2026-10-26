"""
unified_loader.py
==================

Drop-in replacement for run_peb_vs_me.py / run_peb_vs_me_accuracy.py /
run_cross_decoder.py / run_controls.py's `load_sessions()`.

v2 (this package): reads directly from the top-level `data/session_NN.npz`
files (see data/DATA_DICTIONARY.md) instead of maintaining its own
`contrast_fix/unified_session_cache/` intermediate built by combining
masterNeuralStruct_FULL.mat's window spike counts with
masterNeuralStruct_TEST.mat's own trial matrix. That combination step is
no longer needed: `data/session_NN.npz`'s `window_counts` and trial-matrix
fields now come from the *same* FULL-structure session object by
construction, computed in one pass (see data/extract_session.py), so
there is no cross-source trial-count mismatch to guard against -- every
session with a cached window-count/trial-matrix pair is usable here
(the existing downstream "<2-3 neurons per pool" exclusion in
run_peb_vs_me.py etc. still applies exactly as before; sessions 7 and 10
were already excluded that way, for having 0 significantly
right-preferring neurons, independent of any data-availability issue).

Field semantics are unchanged from the previous version: `correct` here
is the trial's raw RESPONSE outcome code equal to 1 (not `choice==CT_X`,
which is a different, equally valid notion of "correct" used elsewhere in
this codebase, e.g. data_io.py / build_real_behavior.py) -- same
trial-inclusion filter (RESPONSE in {-2, 1}, finite RT>0, finite
CONFIDENCE) as always.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import compact_data as CD  # noqa: E402


def load_sessions_unified(min_trials=100, data_dir=None):
    data_dir = data_dir or CD.DEFAULT_DATA_DIR
    sess = []
    for idx in CD.session_indices(data_dir):
        raw = CD.load_raw(idx, data_dir)
        n = int(raw["n"])
        spk = raw["window_counts"]
        side = CD._col(raw, "SIDE", n)
        confidence = CD._col(raw, "CONFIDENCE", n)
        response = CD._col(raw, "RESPONSE", n)
        rt = CD._col(raw, "REACTION_TIME", n)
        contrast_id = CD._col(raw, "CONTRAST_IND", n)
        orientation = CD._col(raw, "ORIENTATION", n)

        # Identical trial-inclusion filter to the original TEST-based
        # pipeline: a defined response code, finite positive RT, finite
        # confidence.
        keep = (np.isin(response, [-2, 1]) & np.isfinite(rt) & (rt > 0)
                & np.isfinite(confidence))
        if keep.sum() < min_trials:
            continue

        sess.append(dict(
            idx=idx,
            spk=spk[keep, :].astype(float),
            signed_ori=orientation[keep].astype(float),
            contrast_id=contrast_id[keep].astype(int),
            conf=confidence[keep].astype(int),
            correct=(response[keep] == 1).astype(int),
            choice=side[keep].astype(int),
            right_pref=raw["right_pref"],
            left_pref=raw["left_pref"],
        ))
    return sess
