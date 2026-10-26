# `data/` — compact per-session dataset

This folder replaces the two large raw MATLAB structures
(`masterNeuralStruct_FULL.mat`, 2.3 GB, and `masterNeuralStruct_TEST.mat`,
201 MB) and the small `masterBehaviorStruct.mat` (11 MB) that earlier
versions of this package required for full reproduction. Everything every
figure/analysis script needs is here, extracted once, directly from
`masterNeuralStruct_FULL.mat` (confirmed a strict superset of `_TEST.mat`:
same 31-column trial matrix schema, plus raw spike times `_TEST.mat` lacks)
by `extract_session.py` (included below for provenance/regeneration if you
ever do have access to the raw file again).

Total size: ~108 MB for 29 sessions (dominated by the per-trial,
per-neuron, per-bin `timecourse_counts` array; still vs. 2.3 GB + 201 MB +
11 MB raw).

## Files

- `session_00.npz` … `session_28.npz` — one file per recording session
  (0-indexed, matching `masterNeuralStruct`'s own session order).
- `date_to_animal.json` — `{experiment date string: "ziggy"/"friedrich"}`,
  extracted once from `masterBehaviorStruct.mat`'s `.observer` field (the
  neural structure's `expt.date` has no animal-identity field of its own).
- `DATA_DICTIONARY.md` — this file.
- `extract_session.py` — the exact extraction script run against
  `masterNeuralStruct_FULL.mat` to produce every `session_NN.npz` (kept for
  provenance; not needed to use this package).

## Each `session_NN.npz`

| Key | Shape | Meaning |
|---|---|---|
| `idx` | scalar | session index (0-based) |
| `expt_date` | string | experiment date, joins to `date_to_animal.json` |
| `trial_matrix` | `(31, n_trials_beh)` | raw `behavior.trialMatrix.trialMatrix`, verbatim. Row `r` for field `name` is `trial_matrix[trial_matrix_index_rows1based[i]-1, :]` where `trial_matrix_index_names[i] == name` |
| `trial_matrix_all` | `(31, n_trials_all)` or `(0,0)` | raw `behavior.trialMatrix.trialMatrix_all`, verbatim, only present for sessions 9–28 (see `has_trial_matrix_all`); this is the *unfiltered* super-set of trials (including aborted/incomplete ones) that `build_real_behavior.py`/Fig. 1's `data_behav.mat` filter down using column 31 (`RESPONSE` code) |
| `has_trial_matrix_all` | bool | whether `trial_matrix_all` is meaningful for this session |
| `trialmatrix_description` | `(31,)` strings | human-readable description of each of the 31 trial-matrix rows, verbatim from the raw file |
| `trial_matrix_index_names`, `trial_matrix_index_rows1based` | `(31,)` | the raw file's own name→row lookup (1-based, as in the original MATLAB/HDF5 structure); identical across all 29 sessions, kept per-session for exact fidelity |
| `valid_idx` | `(n,)` int32 | 0-based indices into the *raw neural* trial dimension (`dataStructTask.unit_spike_times`'s trial axis) used to compute `window_counts`/`timecourse_counts` below — the stimulus-onset-valid trials (`~isnan(stimulus_onsets)`), truncated to `n = min(n_valid_stim, trial_matrix.shape[1])`, exactly matching the trial/neuron reconciliation already used throughout the analysis code (`data_io.py`'s `load_session`/`load_session_timecourse`) |
| `n` | scalar | the reconciled trial count described above; `window_counts`/`timecourse_counts` have this many rows, and the first `n` columns of `trial_matrix` (not `valid_idx`-selected — the behavioral trial matrix is taken by raw position, matching the existing convention) line up with them trial-for-trial |
| `window_counts` | `(n, n_neurons)` float32 | spike count per neuron per trial in the fixed window **[0.050 s, 0.500 s] inclusive of both ends**, post-stimulus-onset — the window used by every main-text pool-fit/gain-map/decoder analysis |
| `timecourse_counts` | `(n, n_neurons, 35)` int16 | spike count per neuron per trial in 20 ms bins, `bin_edges` below (−100 ms to 600 ms) — lets you reconstruct any PSTH/time-resolved analysis; computed in the *same* pass over each neuron/trial's spike times as `window_counts` (so no extra cost), but **not** a re-binning of `window_counts` — each is computed independently and exactly, since the 20 ms bin edges do not land exactly on 50/500 ms |
| `bin_edges` | `(36,)` float64 | seconds, relative to stimulus onset: `arange(-0.100, 0.601, 0.020)` (see "Known issue, fixed in this version" below for why this is `arange`, not `linspace`) |
| `win_start`, `win_end` | scalars | 0.05, 0.5 (seconds) — the window used for `window_counts` |
| `n_neurons_all` | scalar | total neurons recorded in this session, before truncating to the neurons with a preference label (see below) |
| `right_pref`, `left_pref` | `(n_neurons,)` bool | significantly right-/left-choice-preferring neurons (two-sample t-test, right-choice vs. left-choice trials, z-scored counts in the same `[0.05, 0.5]` s window, BH-FDR q=0.05 across neurons within session) — loaded as-is from `shared/neuron_pref.mat` (single source of truth for preference labels throughout this whole package). As of this version `neuron_pref.mat` holds the **recomputed classification** (418 of 859 neurons selective, 207 right-/CW-preferring, 211 left-/CCW-preferring), produced by literally re-running `compute_neuron_pref.m`'s documented procedure against the current data, rather than the originally-shipped 401-neuron classification (193/208) — see "Neuron-classification provenance" below |
| `p_vals`, `t_stats`, `sig_fdr` | `(n_neurons,)` | the underlying statistics behind `right_pref`/`left_pref`, same source |

`n_neurons` above = `min(len(right_pref in neuron_pref.mat), n_neurons_all)`,
matching the truncation already used everywhere else in this codebase.

## Deriving the things older figure-specific caches used to hold

- **`fig2_gain_maps/data/session_cache/session_NN.npz`** (`counts`,
  `choice`, `confidence`, `correct`, `right_pref`, `left_pref`): now
  regenerated from this folder by `fig2_gain_maps/cache_sessions.py`
  (rewritten to read `data/` instead of the raw `.mat` file — same output
  format, so `make_fig2_gain_maps.py` / `make_reliability_supplement.py`
  / `make_axis_comparison.py` are unchanged).
- **The "unified" session data** used by Fig. 3, Fig. 4A, and the SI
  controls (`fig4_peb_pools/unified_loader.py`'s `load_sessions_unified()`):
  now reads `data/session_NN.npz` directly — no separate intermediate
  cache needed, since it was only ever needed before to reconcile two
  *different* raw sources (`FULL`'s spike counts + `TEST`'s trial matrix);
  here both come from the same session file by construction, so sessions
  are no longer dropped for a cross-source trial-count mismatch (though
  the existing downstream "≥3 neurons per pool" inclusion threshold still
  applies exactly as before).
- **`fig6_7_S1_S2_behavioral_models/data/real_behavior_full.csv`**: now
  regenerated from this folder by the rewritten `build_real_behavior.py`
  (same column schema, including `contrast_id`; same `trialMatrix_all`
  + column-31 filter / plain `trialMatrix` fallback logic as before, now
  reading `trial_matrix_all`/`trial_matrix` straight from `data/`).
- **`fig1_task_and_behavior/data/data_behav.mat`**: now regenerated from
  this folder by the new `fig1_task_and_behavior/build_data_behav.py`,
  using the identical valid-trial rule as `real_behavior_full.csv` (same
  37,166-trial result).
- **`fig4_peb_pools/data/figure1_timecourse.npz` and
  `figure1_activity.csv`** (the chosen-/unchosen-pool PSTH and bar-plot
  aggregates behind Fig. 4 panels B-D): these are now **regenerated by
  producer scripts shipped in `fig4_peb_pools/`** (`cache_sessions.py` +
  `compute_figure1_cached.py` for the bar panel; `cache_sessions_timecourse.py`
  + `compute_figure1_timecourse_cached.py` + `compute_activity.py` for the
  PSTH panels), recovered from the lab machine and ported onto the
  compact `data/` folder. These producer scripts were validated by
  confirming they reproduced the originally-shipped 401-neuron-label
  values to float precision before being trusted to run against the
  recomputed (418-neuron) labels now shipped in this version (see
  "Neuron-classification provenance" below).

  **Known issue, fixed in this version**: the original `timecourse_counts`
  in this folder (extracted by an earlier version of `extract_session.py`)
  used `np.linspace(-0.1, 0.6, 36)` for its 20 ms bin edges, which lands
  on *exactly* 0.0 and 0.5 s (to the bit). The already-validated producer
  pipeline this was ported from uses `np.arange(-0.100, 0.601, 0.020)`,
  which has ~1e-17 floating-point slop at those same two edges. Since a
  non-trivial fraction of recorded spikes land at exactly t=0.0 s
  (stimulus onset) or t=0.5 s (the window-count end), the two conventions
  silently placed those spikes in *different* 20 ms bins — enough to
  measurably bias the baseline-corrected PSTH (confirmed: up to several
  spk/s at the affected bins, before a session-average washes most of it
  out). `extract_session.py`'s `BIN_EDGES` now uses the canonical
  `np.arange` form, and `timecourse_counts`/`bin_edges` in every
  `session_NN.npz` in this folder have been re-extracted with it.
  `window_counts` (the Fig. 1-6 analyses that don't touch panels C/D) was
  never affected — it's a plain `>=`/`<=` sum over `[0.05, 0.5]`, not a
  histogram, so there was no bin-edge convention for the floating-point
  slop to land on.

## Neuron-classification provenance

`shared/neuron_pref.mat` (and every `right_pref`/`left_pref`/`p_vals`/
`t_stats`/`sig_fdr` field derived from it throughout this package) was
regenerated for this version by literally re-running
`compute_neuron_pref.m`'s documented procedure (two-sample $t$-test on
$z$-scored window spike counts, right- vs. left-choice trials, BH-FDR
$q=0.05$ within session) against the current raw data. This produced
418 of 859 neurons classified as choice-selective (207 right-/CW-preferring,
211 left-/CCW-preferring), not the 401 (193/208) originally shipped with
earlier versions of this package — the maximum absolute difference between
the two runs' $t$-statistics is 6.84, too large to be a borderline
FDR/boundary effect. The practical difference is a 4-session swap in which
sessions meet the pool-fit's $\geq 3$-neurons-per-pool inclusion criterion
(sessions 3, 9, 24, 27 drop out; 13, 15, 16, 23 come in; still 18 of 29
sessions either way).

Every figure and statistic in the manuscript that depends on these labels
was recomputed under the new classification and cross-checked against an
independent reimplementation before being adopted; the central
chosen/unchosen pool-weighting asymmetry (confidence and accuracy differ in
how they weight the unchosen pool) is unchanged in direction and
significance. The one qualitative change: confidence's own unchosen-pool
weight, taken on its own, is no longer reliably distinguishable from zero
(one-sample Wilcoxon $p=0.038 \to p=0.187$) — already a narrower, more
marginal claim than the main chosen/unchosen asymmetry, and now one that
confidence data alone cannot establish either way. Nothing behavioral
(Figs. 6, 7 and their supplements) is affected, since the behavioral model
fits never touch neuron preference labels.

If you have independent access to the raw recordings and want to verify
this yourself, `../shared/compute_neuron_pref.m` is the exact,
unmodified procedure; re-running it should reproduce `neuron_pref.mat` as
shipped here.

## Trial-matrix column reference (row, 1-based, as in the raw file)

| Row | Field | Used for |
|---|---|---|
| 10 | `CT_X` | `correct = (choice == CT_X)` where `CT_X != 0` |
| 12 | `CONTRAST_IND` | categorical contrast-level id (`contrast_id`) |
| 15 | `CONTRAST` | raw continuous contrast |
| 16 | `ORIENTATION` | signed stimulus orientation (deg) |
| 28 | `REACTION_TIME` | RT |
| 29 | `SIDE` | choice, −1 = left/CCW, +1 = right/CW |
| 30 | `CONFIDENCE` | 0 = low, 1 = high |
| 31 | `RESPONSE` | trial outcome code; `trial_matrix_all` filtered to `RESPONSE in {-2, 1}` |

(Full per-field descriptions are in each session's own
`trialmatrix_description`, verbatim from the raw file.)
