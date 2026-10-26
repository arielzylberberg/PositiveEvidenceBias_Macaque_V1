# Code and data for "A PEB for confidence in V1" (macaque)

This folder contains the code and data needed to reproduce the figures and
reported statistics in the manuscript, and only those — the broader lab
repository this was extracted from contains many exploratory analyses that
never made it into the paper, and none of that is included here.

Each subfolder is self-contained: run its script(s) from inside that folder
and it will read its own `data/` and write to its own `figures/`. A
top-level `shared/` folder holds the few things more than one figure needs
(plotting style, the neuron-preference labels, a color-palette helper).

## What's new in this package (v5, vs. v4)

`shared/neuron_pref.mat` now holds the **recomputed neuron classification**
(418 of 859 neurons selective, 207 right-/CW-preferring, 211
left-/CCW-preferring) rather than the 401-neuron (193/208) classification
shipped with v4 and earlier. This is not a code change — it's a data
update: re-running `compute_neuron_pref.m`'s documented, unmodified
procedure against the current raw data produces 418 selective neurons, not
401, and this package now ships that result as the canonical
`neuron_pref.mat`. See `data/DATA_DICTIONARY.md`'s "Neuron-classification
provenance" section for the full explanation, including the one
qualitative change this produces in the manuscript (confidence's own
unchosen-pool weight is no longer reliably distinguishable from zero on its
own, though the central chosen/unchosen asymmetry between confidence and
accuracy is unchanged). Every downstream artifact that depends on these
labels — `fig2_gain_maps/`, `fig4_peb_pools/` (including the Fig. 4 B-D
panels recovered in v4), and `si_stimulus_difficulty_and_cv_controls/` —
was regenerated from the updated `neuron_pref.mat` and every number was
cross-checked against the manuscript before this package was finalized.
Nothing else changed: `fig1_task_and_behavior/`, `fig3_cross_decoder/`,
`fig5_simulation/`, and `fig6_7_S1_S2_behavioral_models/` don't depend on
neuron preference labels and are untouched.

## What's new in this package (v4, vs. v3)

This pass is pure data/plumbing, not analysis: every number in the
manuscript is unchanged (verified below), the only thing that changed is
*where the data each script reads comes from*.

- **A single compact `data/` folder (top-level, ~108 MB) now replaces the
  three raw MATLAB structures** this package previously needed external
  access to: `masterNeuralStruct_FULL.mat` (2.3 GB), `_TEST.mat` (201 MB),
  and `masterBehaviorStruct.mat` (11 MB). `data/` holds one compact
  `session_NN.npz` per recording session (29 total), extracted once,
  directly from `FULL` (confirmed a strict superset of `TEST` — same
  31-column trial-matrix schema, plus the raw spike times `TEST` lacks),
  by `data/extract_session.py`. See `data/DATA_DICTIONARY.md` for the full
  schema. **You no longer need access to any raw file to regenerate every
  derived artifact in this package from scratch** — only to re-run
  `data/extract_session.py` itself, which nobody but the original authors
  can do anyway (it needs the raw recordings).
- Every per-figure loader that used to depend on a raw `.mat` file was
  rewritten to read `data/` instead, **with identical output** (checked
  bit-for-bit against the previous version's committed outputs — see
  "Validation" below): `fig2_gain_maps/cache_sessions.py`,
  `fig4_peb_pools/unified_loader.py`,
  `fig6_7_S1_S2_behavioral_models/build_real_behavior.py`, and the new
  `fig1_task_and_behavior/build_data_behav.py` (previously this file was
  only ever regenerated ad hoc from the raw data, never by a committed
  script in this package). `shared/compact_data.py` is the one small
  shared module all four now build on.
- **One genuine (harmless) improvement, not just a reshuffle**: Fig. 3,
  Fig. 4A, and the SI controls' `unified_loader.py` previously had to
  combine *two different* raw sources (FULL's spike counts + TEST's trial
  matrix) and dropped 2 of 29 sessions when their trial counts didn't
  cross-match exactly. Since both now come from the same `session_NN.npz`
  by construction, that cross-source mismatch can't happen, and all 29
  sessions now flow through this loader (those 2 sessions are still
  correctly excluded downstream by the existing "≥3 neurons per pool"
  threshold, exactly as before — e.g. Fig. 4A's own `n = 36`
  session×choice pairs is unchanged). The one visible trace of this is a
  diagnostic line in `results_stimulus_and_cv_controls.txt` ("N sessions
  (basic filters)") going from 27 to 29 — not a number quoted anywhere in
  the manuscript, and every actual reported statistic in that file is
  unchanged.
- `fig4_peb_pools/data/figure1_timecourse.npz` and `figure1_activity.csv`
  (the chosen-/unchosen-pool PSTH and bar aggregates behind Fig. 4 panels
  B-D) are **no longer unregenerable precomputed artifacts**. Their
  original producer scripts (`cache_sessions.py`, `compute_figure1_cached.py`,
  `cache_sessions_timecourse.py`, `compute_figure1_timecourse_cached.py`,
  `compute_activity.py`) existed on the lab machine but had never made it
  into any version of this code tree; they're now in `fig4_peb_pools/`,
  rewritten to read the compact `data/` folder instead of
  `masterNeuralStruct_FULL.mat` directly, and both files are regenerated
  by them (run `cache_sessions.py` then `compute_figure1_cached.py` for
  panel B; `cache_sessions_timecourse.py` then
  `compute_figure1_timecourse_cached.py` for panels C/D). Regenerating
  from the shipped `neuron_pref.mat` labels reproduces the previously-shipped
  files to float precision (checked bit-for-bit on the aggregated PSTH/bar
  values; see `data/DATA_DICTIONARY.md` for one data-quality fix this
  turned up along the way). `cache_sessions_timecourse.py`'s own
  intermediate cache (`fig4_peb_pools/data/session_cache_tc/`, ~70 MB
  uncompressed) is **not** included in this package to keep it small —
  it regenerates from the compact `data/` folder in well under a second,
  so there's nothing to be saved by shipping it.

### Validation

Every rewritten loader's output was diffed against this package's own
previous (v3) committed outputs before this package was finalized:
`peb_vs_me.csv`, `peb_vs_me_accuracy.csv`, `cross_decoder_pairs.mat`,
`real_behavior_full.csv`, and `data_behav.mat` are all byte-for-byte
identical; `session_cache/*.npz` (all 29 sessions) matched exactly;
`results_stimulus_and_cv_controls.txt` matched exactly except the one
diagnostic line described above; and every figure script
(`make_fig2_gain_maps.py`, `make_axis_comparison.py`,
`make_reliability_supplement.py`, `run_peb_vs_me.py`,
`run_peb_vs_me_accuracy.py`, `run_cross_decoder.py`, `run_controls.py`,
`make_panels_bcd.py`, `make_figure4_composite.py`,
`recompute_aic_bic.py`/`make_figure6_aic_bic.py`) was re-run end to end
and printed the same headline numbers already quoted in the manuscript
(Fig. 1 n=37,166 trials; Fig. 2 18/29 sessions, 22,034 trials, split-half
reliability 0.49; Fig. 3 AUC 0.709/0.653; Fig. 4A/accuracy n=36
session×choice pairs; Best model by corrected BIC: PositiveEvidence/Gain
for both animals).

## What's new in this package (v3, vs. v2)

Unlike v2, this pass changed real numbers, not just presentation. It was
prompted by an external reproducibility check (a co-author independently
re-ran the analysis pipeline against the raw data and found several
places where the code's actual behavior didn't match what the manuscript
said it did, and one place — Fig. 1's contrast split — where the
manuscript's own numbers were computed on the wrong trial grouping).
Every item below is also described, with its effect on the reported
numbers, in the manuscript's own Methods and Supplemental Information;
this section only summarizes what changed in the code/data here.

- **Contrast split corrected.** Figs. 1, 6, 7, S1, S2 and the SI controls
  previously split trials into "high"/"low" contrast by a median split of
  the raw contrast value. The trial matrix actually encodes contrast as a
  true two-level categorical index (the raw values fall into two clearly
  separated bands, not a continuum), and a median split put a meaningful
  fraction of trials (about 15%) in the wrong group. Every affected
  script now uses the categorical `contrast_id` field directly; old
  median-split outputs are not retained in this package (the manuscript
  documents the specific numbers that changed).
- **Fig. 3, Fig. 4A, and the SI stimulus/cross-validation controls now use
  the same unified, 50–500 ms FULL-window spike-count basis as Fig. 2 and
  Fig. 4B**, instead of a separately precomputed per-trial spike-count
  field (`stimulus_spikeCounts` from `masterNeuralStruct_TEST.mat`) that
  was aligned to stimulus presentation but not numerically identical to
  the stated 50–500 ms window and used a slightly different
  trial-inclusion rule. `fig4_peb_pools/unified_loader.py` builds this
  shared cache (see its module docstring); `run_peb_vs_me.py`,
  `run_peb_vs_me_accuracy.py`, and `run_controls.py` still contain their
  original TEST-struct-based `load_sessions()` function, now unused by
  `main()`, kept only because `unified_loader.py`'s docstring references
  it to explain exactly what changed. This cache covers 27 of the 29
  sessions (2 have zero significantly right-preferring neurons at any
  threshold and were already excluded everywhere else).
- **Fig. 4A's minimum-neurons-per-pool threshold raised from 2 to 3**, to
  match the threshold already used for the gain-map and chosen/unchosen
  session-level analyses, now that Fig. 4A draws on the same spike-count
  basis as those analyses.
- **Fig. 4A's pool-fit regression clarified, not changed in substance**:
  neurons are z-scored within each session × choice subset, and the pool
  means are *not* further standardized afterward (an earlier description
  was ambiguous on this point); the logistic regression uses `C=10`; and
  each choice side (CW/CCW) is its own session/choice data point, not
  averaged together first.
- **Empty spike trains**: for the choice-selectivity classification step
  only, a trial with an empty recorded spike train is now treated as a
  missing value (excluded from that neuron's mean/SD and from the
  classification test), not as a genuine zero count. Every other analysis
  (gain maps, chosen/unchosen pools) still treats an empty train as a
  real zero, as before.
- **`fig6_7_S1_S2_behavioral_models/model_recovery.py`** is new: a
  model-recovery / confusion-matrix validation (20 synthetic datasets per
  model per animal, refit by the identical pipeline used on real data)
  referenced by name in the manuscript's Methods. `run_recovery_both.sh`
  runs both animals end to end (roughly 1.5–2 hours total on 2 workers);
  `model_recovery_Ziggy.csv` / `model_recovery_Friedrich.csv` are the
  completed outputs used for the numbers reported in the manuscript.
- Numbers in Figs. 1, 3, 4, 6, 7, S1, S2 and the SI controls changed as a
  result of the above; figure *appearance* (palette, layout) did not
  change again in this pass beyond what v2 already did.

## What's new in this package (v2, vs. the earlier `code_for_sharing`)

This is a figure-consistency and packaging pass on top of the original
`code_for_sharing`. No analysis, fit, or statistic changed — every script
below was re-run and its printed numbers/stats matched the original
package exactly. What changed is purely presentational:

- **`shared/pub_style.py` / `shared/pub_style.m`** are now the single
  source of truth for font sizes and colors across every figure script
  (previously, five different font-size scales and at least three
  independently-invented "confidence high/low" color schemes were in
  simultaneous use across the figure set). Every Python figure script now
  imports `pub_style` instead of re-declaring its own local copy.
- **Two color pairs were colorblind-unsafe** (checked with a CVD-simulation
  validator, not by eye — see `shared/pub_style.py`'s module docstring for
  the exact Delta-E numbers): the chosen/unchosen-pool colors in Fig. 4/5
  (the paper's central result) and the accuracy correct/incorrect pair.
  Both are fixed; see the docstring for before/after values.
- **Fig. 6** used matplotlib's unthemed default blue/red (the only figure
  that had never been given a deliberate palette) — now uses
  `pub_style`'s `CMP_A`/`CMP_B`.
- **Fig. 7 / S1 / S2** color-code both a contrast split and a confidence
  split in the same 8-panel figure; their hue separation for one pair of
  colors was only borderline under color-vision deficiency, so the
  contrast split now also carries a secondary, non-color encoding (solid
  line + filled marker for high contrast, dashed line + open marker for
  low contrast) on top of its new hue.
- **Figures S3/S4** (the HTML model-overview schematics) are now actually
  rendered to PDF and included in `figS3_S4_model_schematics/figures/` —
  the previous package shipped only the HTML source because no
  Node.js/Playwright was available when it was built. Per guidance from
  the manuscript's author, these two schematics were deliberately **not**
  restyled to match `pub_style` — they're a different kind of figure
  (diagrammatic, not data) and don't need to match the data figures'
  palette.
- **Fig. 1 panels B, C, and D are now all produced by one Python script,
  `fig1_task_and_behavior/make_panels_bcd.py`**, removing the MATLAB/Octave
  dependency entirely. An earlier revision of this package hand-edited
  `run_make_figure.m` (MATLAB, panels B/C) to call `pub_style()` but
  couldn't run or visually verify it (no MATLAB/Octave in the environment
  that built the package); rather than ship an unverified MATLAB edit,
  B and C were rewritten directly in Python, replicating
  `run_make_figure.m`'s logistic-fit/binning logic exactly (same ~5
  quantile bin edges, same no-intercept-vs-intercept choice per panel,
  same population-SEM) — cross-checked to reproduce identical fitted
  coefficients to 4 decimal places. Panel D (previously a separate script,
  `make_panel_d_final.py`, which extracted its own one-off colors/fonts
  from the existing hand-assembled `task_and_behavior.pdf` composite
  rather than using the shared style) is folded into the same script and
  now uses `pub_style.CONF_HIGH`/`CONF_LOW` like every other confidence
  panel in the paper, instead of its old independent green/red. All three
  panels share one data load and one set of bin edges, so they're
  guaranteed consistent with each other, not just separately
  `pub_style`-colored. `run_make_figure.m` is kept in the folder only for
  provenance/cross-checking and is marked superseded in its own header —
  it is no longer part of the recommended path to reproduce Fig. 1.
  Each of the three panels is still saved as its own PDF at the same
  325×275 pt size the old panel D targeted, so each can drop into the
  existing hand-assembled composite layout in place of the old renders;
  a combined 3-up preview image is also included for a quick visual
  check, but that combined file is not itself compositing-ready (panel A
  is still a separate, hand-drawn schematic with no source file).

Every other figure (2, 2-supp-1, 2-supp-2, 3, 4, 5) picked up the new
`pub_style` palette/fonts automatically (they already imported it by
name) or needed only a mechanical import swap, and was verified by
pixel-diffing against the pre-restyle output: only the intended
colors/font sizes changed, nothing else moved.

## Setup

Python 3.10+, packages in `requirements.txt` (`pip install -r
requirements.txt`). Figures S3/S4 need Node.js with the `playwright` npm
package (`npm install playwright && npx playwright install chromium`) —
confirmed working while preparing this package. Everything else, including
all of Fig. 1 now, is plain Python and was verified to run end-to-end
while preparing this package. MATLAB/Octave is no longer required for
anything in this package; `run_make_figure.m` is kept only as an optional,
superseded cross-check (see "What's new" above).

## Figure-by-figure guide

| Manuscript figure | Folder | Run | Produces | Notes |
|---|---|---|---|---|
| Fig. 1 (task & behavior) | `fig1_task_and_behavior/` | `python3 make_panels_bcd.py` | `figures/panelB_psychometric.pdf`, `figures/panelC_confidence.pdf`, `figures/panelD_final.pdf` (+ `figures/panels_BCD_preview.pdf`, a QA-only combined view) | Panel A is a hand-drawn task schematic — no source file exists for it anywhere in the lab repo; the final 4-panel composite PDF is still assembled by hand (Illustrator/Preview) from panel A plus these three. Ran and verified in this v2 package: panel B/C's fitted logistic coefficients were cross-checked to match the retired `run_make_figure.m` (MATLAB) exactly. See "What's new" above for the full story. |
| Fig. 2 (gain maps) | `fig2_gain_maps/` | `python3 make_fig2_gain_maps.py` | `figures/fig2_gain_maps.pdf` | Ran successfully, byte-for-byte the source of the published figure. |
| Fig. 2 — supp. 1 (axis comparison) | `fig2_gain_maps/` | `python3 make_axis_comparison.py` | `figures/fig2_axis_comparison.pdf` | This is `fig2_supp1_axis_comparison.pdf` in the manuscript. |
| Fig. 2 — supp. 2 (d reliability) | `fig2_gain_maps/` | `python3 make_reliability_supplement.py` | `figures/figS_d_reliability.pdf` | This is `fig2_supp2_d_reliability.pdf` in the manuscript. |
| Fig. 3 (cross-decoder) | `fig3_cross_decoder/` | `python3 make_pub_figure.py` | `figures/cross_decoder_pub.pdf` | |
| Fig. 4 (pool activity / PEB) | `fig4_peb_pools/` | `python3 make_figure4_composite.py` | `figures/figure4_peb_composite.pdf` | Imports a few helper functions from `make_figure1.py` (import-safe; no data is read at import time). |
| Fig. 5 (population simulation) | `fig5_simulation/` | `python3 sim_peb_vs_me.py` | `figures/sim_peb_vs_me_composite.pdf` | This is `figure5_sim_peb.pdf` in the manuscript. Pure simulation — no data files needed at all. |
| Fig. 6 (AIC/BIC comparison) | `fig6_7_S1_S2_behavioral_models/` | `python3 recompute_aic_bic.py` then `python3 make_figure6_aic_bic.py` | `figures/figure6_aic_bic.pdf` | `recompute_aic_bic.py` always refits both animals itself (~2–2.5 min total) and does **not** read `model_fits.pkl` for Fig. 6's own numbers — that file is a byproduct of running `fit_free_param_models.py` directly (its own `__main__`, which also writes the uncorrected `model_comparison_aic_bic.csv`). `model_fits.pkl` *is* read by `model_recovery.py` below, as the real-data parameters used to generate synthetic data, so it is needed for that analysis even though not for Fig. 6 itself. `make_figure6_aic_bic.py` itself is fast — it only reads the already-written `model_comparison_aic_bic_corrected.csv`, no refitting. |
| Fig. 7 (best-fit model diagnostics) | `fig6_7_S1_S2_behavioral_models/` | `python3 make_figure7_partial_pos_fit.py` | `figures/figure7_partial_pos_fit.pdf` | |
| Fig. S1, S2 (remaining model fits) | `fig6_7_S1_S2_behavioral_models/` | `python3 make_supp_model_fit_figures.py` | `figures/figureS1_model_fits_1.pdf`, `figureS2_model_fits_2.pdf` | |
| Methods: model-recovery validation | `fig6_7_S1_S2_behavioral_models/` | `python3 fit_free_param_models.py` (writes `model_fits.pkl`) then `./run_recovery_both.sh` (or `python3 model_recovery.py --animal Ziggy/Friedrich --n_replicates 20 --workers N`) | `model_recovery_Ziggy.csv`, `model_recovery_Friedrich.csv` | Slow (~1.5–2 hours total on 2 workers); safe to interrupt and resume, since already-completed (generating model, replicate) pairs in the output CSV are skipped on restart. Use `model_recovery.confusion_matrix()` on the combined CSVs to reproduce the confusion-matrix numbers quoted in Methods. |
| Fig. S3 (model overview schematic) | `figS3_S4_model_schematics/` | `node render_html.js fig_models_overview.html figures/figS3_models_overview.pdf` | `figures/figS3_models_overview.pdf` | **Not previously version-controlled anywhere** — these HTML sources and the render script existed only outside the repo until now. Rendered and verified in this v2 package (Node + Playwright + Chromium were available); the PDF is included in `figures/`. Deliberately not restyled to `pub_style` — see "What's new" above. |
| Fig. S4 (read-outs overview schematic) | `figS3_S4_model_schematics/` | `node render_html.js fig_readouts_overview.html figures/figS4_readouts_overview.pdf` | `figures/figS4_readouts_overview.pdf` | Same as S3 — rendered, verified, and included. |
| §"Why PositiveEvidence/Gain's discount and gain accounts are not distinguishable" (α non-identifiability) | `fig6_7_S1_S2_behavioral_models/` | `python3 diagnose_alpha_unconstrained.py` | printed diagnostic (no figure) | Referenced by name in the manuscript text. |
| SI: stimulus-difficulty confound & cross-validated pool-weighting comparison | `si_stimulus_difficulty_and_cv_controls/` | `python3 run_controls.py` | `results_stimulus_and_cv_controls.txt` | Every statistic quoted in that SI section was checked against this script's output and matches exactly. Reads `data/` via `fig4_peb_pools/unified_loader.py` — no raw data needed (see "Data included vs. data you'd need separately"). |

## Data included vs. data you'd need separately

**As of v4, nothing.** Every figure and statistic in the manuscript can be
reproduced from the ~108 MB top-level `data/` folder included in this
package — no raw `.mat` file of any size is needed. (The three raw
structures this package used to require, `masterNeuralStruct_FULL.mat`
[2.3 GB], `masterNeuralStruct_TEST.mat` [201 MB], and
`masterBehaviorStruct.mat` [11 MB], were only ever the *source* `data/`
was extracted from — see `data/DATA_DICTIONARY.md` and "What's new in
this package (v4)" above. `data/extract_session.py` is kept for
provenance in case you ever need to re-extract from the raw files again,
e.g. with a different spike-counting window, but running it isn't part of
the normal path to reproduce anything in this package.)

`run_cross_decoder.py`, `run_peb_vs_me.py`, `run_peb_vs_me_accuracy.py`,
and `run_controls.py` each still contain their original, now-unused
TEST-struct-based loader function, and `fig2_gain_maps/data_io.py` still
contains the original FULL-struct-based extraction logic — both kept
purely for provenance (so the full data lineage from the raw recordings
is auditable), not because anything still calls them.

## A note on `model_comparison_aic_bic.csv` vs `model_comparison_aic_bic_corrected.csv`

These come from two different scripts, not one: `fit_free_param_models.py`
(run directly) writes the plain `model_comparison_aic_bic.csv` (plus
`model_fits.pkl`); `recompute_aic_bic.py` writes
`model_comparison_aic_bic_corrected.csv` by refitting both animals itself
and re-deriving the pure, unpenalized negative log-likelihood for each
model. Only the `_corrected.csv` version is what Fig. 6 and the reported
AIC/BIC numbers actually use — the plain file and `model_fits.pkl` are
byproducts of running `fit_free_param_models.py` on its own and can be
ignored for reproducing Fig. 6 or its reported numbers. `model_fits.pkl`
is, however, the required input to `model_recovery.py` (see the
figure-by-figure table above), which reads each model's fitted parameters
from it as ground truth for the model-recovery simulation.
