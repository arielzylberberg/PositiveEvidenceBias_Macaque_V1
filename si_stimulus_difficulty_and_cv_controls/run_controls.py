#!/usr/bin/env python3
"""
Stimulus-difficulty controls + cross-validated pool-weighting comparison
for the chosen/unchosen pool-weighting asymmetry (sec:poolfit-methods).

This is an independent re-implementation, run directly against the real
data (masterNeuralStruct_TEST.mat + neuron_pref.mat), NOT a transcription
of numbers reported by a collaborator. It reuses the exact per-session,
per-choice E+/E- construction and inclusion criteria already used for the
published pool-fit result (analysis_az/residual_axis_confidence/
run_peb_vs_me.py and .../residual_axis_accuracy/run_peb_vs_me_accuracy.py,
and sec:poolfit-methods in main5.tex: >=2 neurons/pool, >=80 trials,
>=10 trials/class), so the n and baseline (a,b) here should match the
published pool-fit numbers as a sanity check before trusting the new
control analyses built on top of it.

Three analyses:

  1. BASELINE REPLICATION
     Re-derive a_conf,b_conf,a_acc,b_acc per session x choice pair,
     confirm against the published a~+1.03/-0.28 (conf), a~+1.21/-0.74
     (acc).

  2. STIMULUS-DIFFICULTY CONTROL, residualization (primary; keeps the
     same n=50-ish pair-level unit as the main analysis)
     Within each pair, regress E+ and E- on |orientation| and contrast
     (OLS, with a contrast x |ori| interaction and squared |ori| term to
     absorb a nonlinear/gain-like dependence), take residuals, refit
     confidence~a.E+_res+b.E-_res and accuracy~a.E+_res+b.E-_res, and
     paired-test (b_conf_res - b_acc_res) across pairs (Wilcoxon
     signed-rank, matching the convention already used for the main
     pool-fit's own paired b-test).

  3. STIMULUS-DIFFICULTY CONTROL, matched cells (secondary,
     confirmatory)
     Bin every trial (pooled across sessions, so bin edges are shared)
     into tertiles of |orientation| crossed with the discrete contrast
     level. Within each (session, choice, ori-tertile, contrast) cell
     with enough trials, fit confidence and accuracy models on that
     cell's trials alone (difficulty is ~fixed within a cell by
     construction) and paired-test (b_conf_cell - b_acc_cell) across
     cells.

  4. CROSS-VALIDATED POOL-WEIGHTING COMPARISON
     Pool E+/E- and the confidence label across all qualifying pairs.
     Grouped 5-fold CV by SESSION (not trial), to prevent trials from
     the same session/pair leaking across train/test. In each fold,
     fit three logistic models on the training trials -- full discount
     (confidence ~ E+ alone, i.e. b=0), forced symmetric (confidence ~
     E+-E- alone, i.e. b=-a), and free (confidence ~ E+, E- separately)
     -- and evaluate mean held-out log-likelihood per trial. Report the
     per-fold values (n=5) descriptively; with only 5 folds we do not
     claim formal significance for "X of 5 folds" comparisons.

Outputs a plain-text summary (results_stimulus_and_cv_controls.txt) and
a CSV per analysis, so every number quoted in the manuscript traces back
to a file here.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fig4_peb_pools'))
from unified_loader import load_sessions_unified  # noqa: E402

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'Boundy-Singer_PNAS2024', 'Structures',
                     'masterNeuralStruct_TEST.mat')
PREF = os.path.join(HERE, '..', 'shared', 'neuron_pref.mat')
OUTDIR = HERE
os.makedirs(os.path.join(OUTDIR, 'data'), exist_ok=True)

COL_CONT_IND   = 12 - 1
COL_ORIENT     = 16 - 1
COL_RT         = 28 - 1
COL_CHOICE     = 29 - 1
COL_CONFIDENCE = 30 - 1
COL_RESPONSE   = 31 - 1

MIN_TRIALS_PAIR = 80
MIN_CLASS_COUNT = 10
# Raised from 2 to 3 to match the gain-map/chosen-unchosen threshold, now
# that this script shares the same FULL-window spike-count basis as those
# analyses (see run_peb_vs_me.py and main7.tex's sec:poolfit-methods).
MIN_POOL_NEURONS = 3

RNG_SEED = 0


# --------------------------------------------------------------------
# Data loading: now the unified FULL-window basis (see
# fig4_peb_pools/unified_loader.py) instead of TEST's stimulus_spikeCounts.
# --------------------------------------------------------------------
def load_sessions(path):
    return load_sessions_unified()


def load_pref():
    d = sio.loadmat(PREF, squeeze_me=True, struct_as_record=False)
    arr = d['neuron_pref_data']
    return [dict(left=np.asarray(a.left_pref).astype(bool),
                 right=np.asarray(a.right_pref).astype(bool),
                 sig=np.asarray(a.sig_fdr).astype(bool))
            for a in arr]


# --------------------------------------------------------------------
# Build one row per qualifying (session, choice) pair with everything
# needed downstream: trial-level E+, E-, |ori|, contrast, confidence,
# correct.
# --------------------------------------------------------------------
def build_pairs(sess, prefs):
    pairs = []
    for s in sess:
        pref = prefs[s['idx']]
        cw_pref = pref['right'] & pref['sig']
        ccw_pref = pref['left'] & pref['sig']
        if cw_pref.sum() < MIN_POOL_NEURONS or ccw_pref.sum() < MIN_POOL_NEURONS:
            continue
        for ch in (+1, -1):
            mask = s['choice'] == ch
            if mask.sum() < MIN_TRIALS_PAIR:
                continue
            y_conf = s['conf'][mask]
            y_acc = s['correct'][mask]
            if (min(np.bincount(y_conf, minlength=2)) < MIN_CLASS_COUNT or
                    min(np.bincount(y_acc, minlength=2)) < MIN_CLASS_COUNT):
                continue
            Xc = s['spk'][mask, :]
            sc = StandardScaler().fit(Xc)
            Xz = sc.transform(Xc)
            if ch == +1:
                chosen_pool, nonchosen_pool = cw_pref, ccw_pref
            else:
                chosen_pool, nonchosen_pool = ccw_pref, cw_pref
            E_plus = Xz[:, chosen_pool].mean(axis=1)
            E_minus = Xz[:, nonchosen_pool].mean(axis=1)
            pairs.append(dict(
                session=s['idx'], choice=ch, n=int(mask.sum()),
                E_plus=E_plus, E_minus=E_minus,
                abs_ori=np.abs(s['signed_ori'][mask]),
                contrast=s['contrast_id'][mask],
                conf=y_conf, correct=y_acc))
    return pairs


def fit_ab(X, y):
    """confidence/accuracy ~ a.E+ + b.E- , same regularization as the
    published pool-fit (C=10, matches run_peb_vs_me*.py)."""
    if np.unique(y).size < 2:
        return None
    clf = LogisticRegression(C=10.0, max_iter=1000).fit(X, y)
    a, b = clf.coef_.ravel()
    return float(a), float(b)


# ======================================================================
# 1. BASELINE REPLICATION
# ======================================================================
def analysis_baseline(pairs):
    rows = []
    for p in pairs:
        X = np.column_stack([p['E_plus'], p['E_minus']])
        r_conf = fit_ab(X, p['conf'])
        r_acc = fit_ab(X, p['correct'])
        if r_conf is None or r_acc is None:
            continue
        rows.append(dict(session=p['session'], choice=p['choice'], n=p['n'],
                          a_conf=r_conf[0], b_conf=r_conf[1],
                          a_acc=r_acc[0], b_acc=r_acc[1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'data', 'baseline_replication.csv'), index=False)
    out = {'n_pairs': len(df)}
    for col in ('a_conf', 'b_conf', 'a_acc', 'b_acc'):
        out[f'{col}_mean'] = df[col].mean()
        out[f'{col}_median'] = df[col].median()
    out['b_gap_mean'] = (df.b_conf - df.b_acc).mean()
    out['b_gap_p'] = stats.wilcoxon(df.b_conf - df.b_acc).pvalue
    out['b_conf_vs0_p'] = stats.wilcoxon(df.b_conf).pvalue
    return df, out


# ======================================================================
# 2. RESIDUALIZATION CONTROL
# ======================================================================
def _residualize(y, abs_ori, contrast):
    """OLS residual of y against a flexible stimulus-difficulty basis:
    |ori|, |ori|^2, contrast dummies, and |ori| x contrast interaction.
    Any contrast/orientation-driven (i.e. difficulty-driven) component
    of y is removed; only the residual, difficulty-independent part
    survives."""
    contrast = pd.Categorical(contrast)
    Xc = pd.get_dummies(contrast, drop_first=True).to_numpy(dtype=float)
    ori = abs_ori.reshape(-1, 1)
    ori2 = ori ** 2
    inter = Xc * ori  # interaction of each non-reference contrast dummy with |ori|
    design = np.column_stack([np.ones(len(y)), ori, ori2, Xc, inter])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def analysis_residualized(pairs):
    rows = []
    for p in pairs:
        if len(np.unique(p['contrast'])) < 2:
            # residualizing against a single contrast level is degenerate;
            # still residualize against |ori| alone in that case
            contrast_for_resid = np.zeros_like(p['contrast'])
        else:
            contrast_for_resid = p['contrast']
        Ep_res = _residualize(p['E_plus'], p['abs_ori'], contrast_for_resid)
        Em_res = _residualize(p['E_minus'], p['abs_ori'], contrast_for_resid)
        X = np.column_stack([Ep_res, Em_res])
        r_conf = fit_ab(X, p['conf'])
        r_acc = fit_ab(X, p['correct'])
        if r_conf is None or r_acc is None:
            continue
        rows.append(dict(session=p['session'], choice=p['choice'], n=p['n'],
                          a_conf=r_conf[0], b_conf=r_conf[1],
                          a_acc=r_acc[0], b_acc=r_acc[1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'data', 'residualized_control.csv'), index=False)
    out = {'n_pairs': len(df)}
    out['b_gap_mean'] = (df.b_conf - df.b_acc).mean()
    out['b_gap_median'] = (df.b_conf - df.b_acc).median()
    out['b_gap_p'] = stats.wilcoxon(df.b_conf - df.b_acc).pvalue
    out['n_wins'] = int((df.b_conf > df.b_acc).sum())
    out['b_conf_mean'] = df.b_conf.mean()
    out['b_acc_mean'] = df.b_acc.mean()
    out['b_conf_vs0_p'] = stats.wilcoxon(df.b_conf).pvalue
    return df, out


# ======================================================================
# 3. MATCHED-CELL CONTROL
# ======================================================================
def analysis_matched_cells(pairs, n_ori_bins=3,
                            min_cell_trials=30, min_cell_class=5):
    all_ori = np.concatenate([p['abs_ori'] for p in pairs])
    edges = np.quantile(all_ori, np.linspace(0, 1, n_ori_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    rows = []
    for p in pairs:
        ori_bin = np.digitize(p['abs_ori'], edges) - 1
        ori_bin = np.clip(ori_bin, 0, n_ori_bins - 1)
        for ob in range(n_ori_bins):
            for cval in np.unique(p['contrast']):
                m = (ori_bin == ob) & (p['contrast'] == cval)
                if m.sum() < min_cell_trials:
                    continue
                yc, ya = p['conf'][m], p['correct'][m]
                if (min(np.bincount(yc, minlength=2)) < min_cell_class or
                        min(np.bincount(ya, minlength=2)) < min_cell_class):
                    continue
                X = np.column_stack([p['E_plus'][m], p['E_minus'][m]])
                r_conf = fit_ab(X, yc)
                r_acc = fit_ab(X, ya)
                if r_conf is None or r_acc is None:
                    continue
                rows.append(dict(session=p['session'], choice=p['choice'],
                                  ori_bin=ob, contrast=cval, n=int(m.sum()),
                                  a_conf=r_conf[0], b_conf=r_conf[1],
                                  a_acc=r_acc[0], b_acc=r_acc[1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'data', 'matched_cell_control.csv'), index=False)
    out = {'n_cells': len(df), 'ori_bin_edges': edges.tolist(),
           'n_pairs_contributing': df[['session', 'choice']].drop_duplicates().shape[0]
           if len(df) else 0}
    if len(df) >= 2:
        out['b_gap_mean'] = (df.b_conf - df.b_acc).mean()
        out['b_gap_median'] = (df.b_conf - df.b_acc).median()
        out['b_gap_p'] = stats.wilcoxon(df.b_conf - df.b_acc).pvalue
        out['n_wins'] = int((df.b_conf > df.b_acc).sum())
    return df, out


# ======================================================================
# 4. CROSS-VALIDATED POOL-WEIGHTING COMPARISON (session-grouped 5-fold)
# ======================================================================
def analysis_cv(pairs, seed=RNG_SEED):
    E_plus = np.concatenate([p['E_plus'] for p in pairs])
    E_minus = np.concatenate([p['E_minus'] for p in pairs])
    y = np.concatenate([p['conf'] for p in pairs])
    groups = np.concatenate([np.full(p['n'], p['session']) for p in pairs])

    n_groups = len(np.unique(groups))
    # Leave-one-SESSION-out (not an arbitrary small k): maximizes training
    # data per fold, gives one held-out value per session so the
    # discount-vs-symmetric and free-vs-discount comparisons can be
    # paired-tested across sessions (n~25-29) instead of across an
    # arbitrary handful of folds with little power.
    logo = LeaveOneGroupOut()

    DV = E_plus - E_minus
    rows = []
    for fold, (tr, te) in enumerate(logo.split(E_plus, y, groups)):
        if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
            continue
        # full discount: b = 0 -> confidence ~ E+ alone
        clf0 = LogisticRegression(C=10.0, max_iter=1000).fit(E_plus[tr, None], y[tr])
        p0 = clf0.predict_proba(E_plus[te, None])[:, 1]
        ll0 = -log_loss(y[te], p0, labels=[0, 1])

        # forced symmetric: b = -a -> confidence ~ (E+ - E-) alone
        clfs = LogisticRegression(C=10.0, max_iter=1000).fit(DV[tr, None], y[tr])
        ps = clfs.predict_proba(DV[te, None])[:, 1]
        lls = -log_loss(y[te], ps, labels=[0, 1])

        # free: confidence ~ E+, E- separately
        Xtr = np.column_stack([E_plus[tr], E_minus[tr]])
        Xte = np.column_stack([E_plus[te], E_minus[te]])
        clff = LogisticRegression(C=10.0, max_iter=1000).fit(Xtr, y[tr])
        pf = clff.predict_proba(Xte)[:, 1]
        llf = -log_loss(y[te], pf, labels=[0, 1])

        rows.append(dict(fold=fold, session=groups[te][0], n_test=len(te),
                          ll_discount=ll0, ll_symmetric=lls, ll_free=llf))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'data', 'cv_pool_weighting.csv'), index=False)
    # weight each session's LL by its number of held-out trials when
    # computing the overall mean (unweighted mean would let a 100-trial
    # session count as much as a 5000-trial session)
    w = df.n_test.to_numpy(float)
    out = {
        'n_sessions_held_out': len(df), 'n_groups': n_groups,
        'll_discount_mean': np.average(df.ll_discount, weights=w),
        'll_symmetric_mean': np.average(df.ll_symmetric, weights=w),
        'll_free_mean': np.average(df.ll_free, weights=w),
        'free_beats_discount_nsessions': int((df.ll_free > df.ll_discount).sum()),
        'discount_beats_symmetric_nsessions': int((df.ll_discount > df.ll_symmetric).sum()),
    }
    if len(df) >= 2:
        out['discount_vs_symmetric_p'] = stats.wilcoxon(
            df.ll_discount - df.ll_symmetric).pvalue
        out['free_vs_discount_p'] = stats.wilcoxon(
            df.ll_free - df.ll_discount).pvalue
    return df, out


# ======================================================================
def main():
    print('Loading sessions...', flush=True)
    sess = load_sessions(DATA)
    print(f'  {len(sess)} sessions pass basic trial filters', flush=True)
    prefs = load_pref()
    pairs = build_pairs(sess, prefs)
    print(f'  {len(pairs)} (session x choice) pairs pass inclusion criteria', flush=True)

    report = []
    report.append(f'N sessions (basic filters): {len(sess)}')
    report.append(f'N (session x choice) pairs (pool-fit inclusion criteria): {len(pairs)}')
    report.append('')

    print('1. Baseline replication...', flush=True)
    df_base, out_base = analysis_baseline(pairs)
    report.append('=== 1. BASELINE REPLICATION (sanity check vs published a,b) ===')
    for k, v in out_base.items():
        report.append(f'  {k}: {v}')
    report.append('')

    print('2. Residualized control...', flush=True)
    df_res, out_res = analysis_residualized(pairs)
    report.append('=== 2. STIMULUS-DIFFICULTY CONTROL: residualized E+/E- ===')
    for k, v in out_res.items():
        report.append(f'  {k}: {v}')
    report.append('')

    print('3. Matched-cell control...', flush=True)
    df_cell, out_cell = analysis_matched_cells(pairs)
    report.append('=== 3. STIMULUS-DIFFICULTY CONTROL: orientation x contrast matched cells ===')
    for k, v in out_cell.items():
        report.append(f'  {k}: {v}')
    report.append('')

    print('4. Cross-validated comparison...', flush=True)
    df_cv, out_cv = analysis_cv(pairs)
    report.append('=== 4. CROSS-VALIDATED POOL-WEIGHTING COMPARISON (session-grouped 5-fold) ===')
    for k, v in out_cv.items():
        report.append(f'  {k}: {v}')
    report.append('')
    report.append('Per-fold CV table:')
    report.append(df_cv.to_string(index=False))

    text = '\n'.join(str(x) for x in report)
    with open(os.path.join(OUTDIR, 'results_stimulus_and_cv_controls.txt'), 'w') as f:
        f.write(text + '\n')
    print('\n' + text)


if __name__ == '__main__':
    main()
