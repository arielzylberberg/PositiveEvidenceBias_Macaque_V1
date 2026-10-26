"""
compact_data.py
================

Thin, shared reader for the top-level `data/session_NN.npz` files (see
`data/DATA_DICTIONARY.md`) that every figure folder's own loader is built
on top of. This module does no analysis-specific logic itself -- it just
turns the raw per-session npz into convenient arrays, exactly reproducing
the field conventions `data_io.py` / `build_unified_session_data.py` /
`build_real_behavior.py` established when this all still came from the
raw masterNeuralStruct_FULL.mat / _TEST.mat / masterBehaviorStruct.mat.

DATA_DIR defaults to the top-level data/ folder two levels up from any
figure subfolder (<repo_root>/data); override with the
BOUNDY_SINGER_DATA_DIR environment variable if you've moved it.
"""
from __future__ import annotations

import glob
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.environ.get(
    "BOUNDY_SINGER_DATA_DIR",
    os.path.join(_HERE, "..", "data"),
)


def session_files(data_dir: str = DEFAULT_DATA_DIR):
    return sorted(glob.glob(os.path.join(data_dir, "session_*.npz")))


def session_indices(data_dir: str = DEFAULT_DATA_DIR):
    idxs = []
    for fn in session_files(data_dir):
        base = os.path.basename(fn)
        idxs.append(int(base.replace("session_", "").replace(".npz", "")))
    return sorted(idxs)


def load_raw(idx: int, data_dir: str = DEFAULT_DATA_DIR):
    """Return the raw npz contents for one session (NpzFile; index like a dict)."""
    return np.load(os.path.join(data_dir, f"session_{idx:02d}.npz"), allow_pickle=True)


def _col(raw, name: str, n: int | None = None, matrix_key: str = "trial_matrix"):
    """Look up one named trial-matrix column (1-based row lookup stored in
    the npz), optionally truncated to the first n trials -- matching the
    `col(name)` helper in the original data_io.py / build_unified_session_data.py,
    which index the behavioral trial matrix by raw position, not by
    `valid_idx` (that index only applies to the neural/spike-time side)."""
    names = raw["trial_matrix_index_names"].tolist()
    rows1 = raw["trial_matrix_index_rows1based"]
    row = int(rows1[names.index(name)]) - 1
    col = raw[matrix_key][row, :]
    return col[:n] if n is not None else col


def load_session(idx: int, data_dir: str = DEFAULT_DATA_DIR):
    """Drop-in equivalent of data_io.py's `load_session`: window spike
    counts + choice/confidence/correct/contrast/orientation/rt, trial and
    neuron dimensions already reconciled (see DATA_DICTIONARY.md)."""
    raw = load_raw(idx, data_dir)
    n = int(raw["n"])

    choice = _col(raw, "SIDE", n)
    confidence = _col(raw, "CONFIDENCE", n)
    ctx = _col(raw, "CT_X", n)
    contrast_id = _col(raw, "CONTRAST_IND", n)
    contrast = _col(raw, "CONTRAST", n)
    orientation = _col(raw, "ORIENTATION", n)
    rt = _col(raw, "REACTION_TIME", n)

    correct = np.full(n, np.nan)
    has_target = ctx != 0
    correct[has_target] = (choice[has_target] == ctx[has_target]).astype(float)

    return dict(
        idx=idx,
        n_trials=n,
        n_neurons=raw["window_counts"].shape[1],
        counts=raw["window_counts"],
        choice=choice,
        confidence=confidence,
        correct=correct,
        contrast_id=contrast_id,
        contrast=contrast,
        orientation=orientation,
        rt=rt,
        right_pref=raw["right_pref"],
        left_pref=raw["left_pref"],
        win_start=float(raw["win_start"]),
        win_end=float(raw["win_end"]),
    )


def load_session_timecourse(idx: int, data_dir: str = DEFAULT_DATA_DIR):
    """Drop-in equivalent of data_io.py's `load_session_timecourse`:
    20 ms-binned spike counts over the same trial/neuron reconciliation
    as `load_session` above."""
    raw = load_raw(idx, data_dir)
    base = load_session(idx, data_dir)
    base["counts_tc"] = raw["timecourse_counts"]
    base["bin_edges"] = raw["bin_edges"]
    return base


def build_real_behavior_rows(data_dir: str = DEFAULT_DATA_DIR, date_to_animal: dict | None = None):
    """Reproduce build_real_behavior.py / rebuild_with_contrast_id.py's
    row-by-row extraction exactly: `trial_matrix_all` + RESPONSE(col 31)
    in {-2, 1} filter when present (sessions 9-28), else the plain
    `trial_matrix` unfiltered (sessions 0-8) -- same rule, same column
    positions (0-indexed): 9=CT_X("req_resp"), 14=CONTRAST, 15=ORIENTATION,
    11=CONTRAST_IND, 27=REACTION_TIME, 28=SIDE, 29=CONFIDENCE.
    """
    if date_to_animal is None:
        import json
        with open(os.path.join(data_dir, "date_to_animal.json")) as fh:
            date_to_animal = json.load(fh)

    rows = []
    for idx in session_indices(data_dir):
        raw = load_raw(idx, data_dir)
        date = str(raw["expt_date"])
        animal = date_to_animal.get(date)
        if bool(raw["has_trial_matrix_all"]):
            tm = raw["trial_matrix_all"]
            col31 = tm[30, :]
            keep = np.isin(col31, [-2, 1])
            tm = tm[:, keep]
        else:
            tm = raw["trial_matrix"]

        req_resp, contrast, orientation = tm[9, :], tm[14, :], tm[15, :]
        contrast_id = tm[11, :]
        rt, choice, confidence = tm[27, :], tm[28, :], tm[29, :]
        correct = (choice == req_resp).astype(float)
        for j in range(tm.shape[1]):
            rows.append((idx, date, animal, orientation[j], contrast[j], contrast_id[j],
                         correct[j], confidence[j], choice[j], rt[j]))
    return rows
