#!/usr/bin/env python3
"""
Build a clean, real, per-trial behavioral table for fit_free_param_models.py
that has NO dependence on neural data whatsoever.

v2 (this package): reads the top-level ../data/session_NN.npz files (see
../data/DATA_DICTIONARY.md) instead of masterNeuralStruct_TEST.mat +
masterBehaviorStruct.mat directly. Row-for-row identical extraction logic
to the original (same two-branch valid-trial rule, same column positions,
same date->animal join) -- see shared/compact_data.py's
`build_real_behavior_rows()`, which this script now just calls.

Per session, the valid-trial subset is selected with the SAME two-branch
rule used to build data_behav.mat (the source of the manuscript's Fig. 1
trial count): if a session has a `trialMatrix_all` table, use it and keep
rows where column 31 (1-indexed) is in {-2, 1} ("incorrect"/"correct"
codes); otherwise fall back to the plain `trialMatrix` table in full. This
gives the same 37,166 valid trials across the 29 sessions as before.

  column 10  X location of correct choice target/side      -> "req_resp"
  column 12  Contrast-level index (categorical)             -> "contrast_id"
  column 15  Contrast of stimulus (raw, continuous)          -> "contrast"
  column 16  Orientation of stimulus in degrees (signed)      -> "orientation"
  column 28  Reaction time                                   -> "rt"
  column 29  observer choice, left/right                      -> "choice"
  column 30  observer confidence, range [0,1] = [low,high]    -> "confidence"

  correct = (choice == req_resp)

Usage:
    python3 build_real_behavior.py [out_path]
    (defaults to data/real_behavior_full.csv)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import compact_data as CD  # noqa: E402


def extract():
    rows = CD.build_real_behavior_rows()
    df = pd.DataFrame(rows, columns=["session_idx", "date", "animal", "orientation",
                                      "contrast", "contrast_id", "correct", "confidence",
                                      "choice", "rt"])
    return df


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "real_behavior_full.csv")
    df = extract()
    print(f"{len(df)} valid trials, {df.session_idx.nunique()} sessions")
    print(df.animal.value_counts())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
