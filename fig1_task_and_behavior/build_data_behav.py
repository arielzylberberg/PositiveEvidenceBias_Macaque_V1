#!/usr/bin/env python3
"""
Build data/data_behav.mat (the one file behind every Fig. 1 panel: B, C, D)
from the top-level ../data/session_NN.npz files (see
../data/DATA_DICTIONARY.md) -- no raw masterNeuralStruct_*.mat required.

Uses the exact same valid-trial rule as fig6_7_S1_S2_behavioral_models'
real_behavior_full.csv (trialMatrix_all + RESPONSE-code filter where
available, else the plain trialMatrix unfiltered) -- same 37,166 trials
across the 29 sessions -- via shared/compact_data.py's
`build_real_behavior_rows()`, then repackaged into the flat
contrast/orientation/RT/choice/confidence/correct/contrast_id arrays
data_behav.mat has always had.

Usage:
    python3 build_data_behav.py [out_path]
    (defaults to data/data_behav.mat)
"""
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import compact_data as CD  # noqa: E402


def build():
    rows = CD.build_real_behavior_rows()
    # rows: (session_idx, date, animal, orientation, contrast, contrast_id,
    #        correct, confidence, choice, rt)
    orientation = np.array([r[3] for r in rows], dtype=float)
    contrast = np.array([r[4] for r in rows], dtype=float)
    contrast_id = np.array([r[5] for r in rows], dtype=float)
    correct = np.array([r[6] for r in rows], dtype=np.uint8)
    confidence = np.array([r[7] for r in rows], dtype=np.uint8)
    choice = np.array([r[8] for r in rows], dtype=np.int16)
    rt = np.array([r[9] for r in rows], dtype=float)
    return dict(
        contrast=contrast, orientation=orientation, RT=rt, choice=choice,
        confidence=confidence, correct=correct, contrast_id=contrast_id,
    )


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "data_behav.mat")
    d = build()
    print(f"{len(d['contrast'])} valid trials")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sio.savemat(out_path, d)
    print(f"Saved {out_path}")
