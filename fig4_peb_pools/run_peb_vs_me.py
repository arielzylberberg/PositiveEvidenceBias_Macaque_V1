"""
Positive-Evidence Bias  vs  Magnitude Effect
============================================

Decompose confidence (and the orthogonal axis y from the residual-
axis pipeline) into chosen-pool (E⁺) and non-chosen-pool (E⁻)
activity, per session × choice.

    conf  ~  a · E⁺  +  b · E⁻
    y     ~  a_y · E⁺  +  b_y · E⁻

where the chosen vs non-chosen pools are defined by the preference
labels in `../neuron_pref/neuron_pref.mat`, swapped per choice
(so E⁺ is always the pool that matches the choice that was made).

Diagnostic for the (a, b) pair:

  *  pure DV / accuracy readout      a = +c,  b = −c     (sign-flipped)
  *  POSITIVE-EVIDENCE BIAS          a > 0,  b ≈ 0 or < 0,  |a| > |b|
  *  MAGNITUDE / GAIN EFFECT         a = b > 0           (symmetric, both +)
  *  mixed PEB+ME                    a > b > 0           (both +, but a > b)

The diagnostic is the asymmetry index

    α  =  (a − |b|) / (a + |b|)     ∈ [−1, +1]

   α ≈ +1  → b ≈ 0  ⇒ pure PEB (chosen pool only)
   α ≈ 0   → a = |b| ⇒ either pure DV (b<0) or pure ME (b>0).  Split
              by sign of b: b<0 → DV; b>0 → ME.
   α ≈ −1  → a = 0  ⇒ confidence reads only the NON-chosen pool (not
              expected)

We compute one (a, b) pair per (session × choice) and pool across
both choices per session.  Targets: actual confidence label (logistic)
and the y-axis output (linear).
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import mat73
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_loader import load_sessions_unified  # noqa: E402

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'Boundy-Singer_PNAS2024', 'Structures',
                    'masterNeuralStruct_TEST.mat')
PREF = os.path.join(HERE, '..', 'shared', 'neuron_pref.mat')
FIG  = os.path.join(HERE, 'figures');  os.makedirs(FIG, exist_ok=True)

# Minimum neurons required in each preference pool. Raised from 2 to 3 to
# match the threshold used for the gain-map/chosen-unchosen session-level
# analyses (sec:dg-axes-methods, sec:chosen-unchosen-methods) now that this
# script also draws on the same FULL-window spike-count basis as those
# analyses -- see main7.tex's updated sec:poolfit-methods for the rationale.
MIN_POOL_NEURONS = 3

COL_CONT_IND   = 12 - 1
COL_ORIENT     = 16 - 1
COL_RT         = 28 - 1
COL_CHOICE     = 29 - 1
COL_CONFIDENCE = 30 - 1
COL_RESPONSE   = 31 - 1


def load_sessions(path):
    d = mat73.loadmat(path);  mb = d['masterNeuralStruct']
    sess = []
    for i, s in enumerate(mb):
        ori_list = np.array(s['dataStructTask']['trial_ori_list'])
        spk = np.array(s['dataStructTask']['stimulus_spikeCounts'])
        beh = s['behavior']['trialMatrix']
        if 'trialMatrix_all' in beh:
            tma = np.array(beh['trialMatrix_all'])
            keep0 = np.isin(tma[:, COL_RESPONSE], [-2, 1]) & ~np.isnan(ori_list)
            tm = tma[keep0, :]; spk = spk[keep0, :]
        else:
            tm = np.array(beh['trialMatrix'])
            spk = spk[~np.isnan(ori_list), :]
        keep = np.isin(tm[:, COL_RESPONSE], [-2, 1]) \
                & np.isfinite(tm[:, COL_RT]) & (tm[:, COL_RT] > 0) \
                & np.isfinite(tm[:, COL_CONFIDENCE])
        if keep.sum() < 100: continue
        tm = tm[keep, :]; spk = spk[keep, :]
        sess.append(dict(
            idx=i, spk=spk.astype(float),
            signed_ori=tm[:, COL_ORIENT].astype(float),
            contrast_id=tm[:, COL_CONT_IND].astype(int),
            conf=tm[:, COL_CONFIDENCE].astype(int),
            correct=(tm[:, COL_RESPONSE] == 1).astype(int),
            choice=tm[:, COL_CHOICE].astype(int)))
    return sess


def load_pref():
    d = sio.loadmat(PREF, squeeze_me=True, struct_as_record=False)
    arr = d['neuron_pref_data']
    return [dict(left=np.asarray(a.left_pref).astype(bool),
                 right=np.asarray(a.right_pref).astype(bool),
                 sig=np.asarray(a.sig_fdr).astype(bool))
            for a in arr]


def cv_predict_logit(X, y, seed=0, n_splits=5):
    """Held-out conf prediction (probability)."""
    if np.unique(y).size < 2:  return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps);  return np.log(p / (1 - p))


# ----------------------------------------------------------------------
def per_choice_decomp(s, pref):
    """Return dict per choice with E+/E- and (a, b) coefficients."""
    out = {}
    cw_pref  = pref['right'] & pref['sig']
    ccw_pref = pref['left']  & pref['sig']
    if cw_pref.sum() < MIN_POOL_NEURONS or ccw_pref.sum() < MIN_POOL_NEURONS:
        return None
    for ch in (+1, -1):
        mask = s['choice'] == ch
        if mask.sum() < 80:
            out[ch] = None;  continue
        Xc = s['spk'][mask, :]
        y_conf = s['conf'][mask]
        y_acc  = s['correct'][mask]
        if (min(np.bincount(y_conf, minlength=2)) < 10 or
            min(np.bincount(y_acc , minlength=2)) < 10):
            out[ch] = None; continue
        # within-session z-score per neuron (across these subset trials)
        sc = StandardScaler().fit(Xc)
        Xz = sc.transform(Xc)
        # chosen / non-chosen pool means
        if ch == +1:
            chosen_pool, nonchosen_pool = cw_pref, ccw_pref
        else:
            chosen_pool, nonchosen_pool = ccw_pref, cw_pref
        E_plus  = Xz[:, chosen_pool   ].mean(axis=1)
        E_minus = Xz[:, nonchosen_pool].mean(axis=1)

        # ---- compute the y axis  (residual of conf decoder against acc) ----
        p_acc  = cv_predict_logit(Xc, y_acc)
        p_conf = cv_predict_logit(Xc, y_conf)
        if p_acc is None or p_conf is None:
            out[ch] = None; continue
        xl = logit(p_acc);  cl = logit(p_conf)
        A = np.column_stack([np.ones_like(xl), xl])
        beta, *_ = np.linalg.lstsq(A, cl, rcond=None)
        y_resid = cl - A @ beta            # orthogonal axis

        # ---- fit  conf ~ E+ + E-  (logistic) ----
        Xreg = np.column_stack([E_plus, E_minus])
        clf = LogisticRegression(C=10.0, max_iter=500).fit(Xreg, y_conf)
        a_conf, b_conf = clf.coef_.ravel()

        # ---- fit  y_resid ~ E+ + E-  (linear) ----
        X1 = np.column_stack([np.ones_like(E_plus), E_plus, E_minus])
        beta_y, *_ = np.linalg.lstsq(X1, y_resid, rcond=None)
        a_y, b_y = beta_y[1], beta_y[2]

        # ---- fit also  conf ~ DV + Sum  for explicit decomposition ----
        DV  = E_plus - E_minus
        Sum = E_plus + E_minus
        clf2 = LogisticRegression(C=10.0, max_iter=500).fit(
            np.column_stack([DV, Sum]), y_conf)
        beta_DV, beta_Sum = clf2.coef_.ravel()

        out[ch] = dict(
            n=int(mask.sum()),
            a_conf=float(a_conf), b_conf=float(b_conf),
            a_y=float(a_y),       b_y=float(b_y),
            beta_DV=float(beta_DV), beta_Sum=float(beta_Sum))
    return out


# ----------------------------------------------------------------------
def asym(a, b):
    """Asymmetry index = (a − |b|)/(a + |b|).  +1 → pure PEB, 0 → equal."""
    denom = a + abs(b)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return (a - abs(b)) / denom


def main():
    sess = load_sessions_unified()
    prefs = load_pref()
    rows = []
    for s in sess:
        out = per_choice_decomp(s, prefs[s['idx']])
        if out is None: continue
        for ch in (+1, -1):
            r = out[ch]
            if r is None: continue
            rows.append(dict(
                session=s['idx'], choice=('CW' if ch == +1 else 'CCW'),
                **{k: r[k] for k in ('n', 'a_conf', 'b_conf', 'a_y', 'b_y',
                                       'beta_DV', 'beta_Sum')}))
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'{len(df)} (session × choice) rows.')

    # ---------------- summary stats ----------------
    print('\n=== conf ~ a·E⁺ + b·E⁻  (logistic, raw spike z-scores) ===')
    for col, name in [('a_conf', 'a (chosen pool)'),
                       ('b_conf', 'b (non-chosen pool)')]:
        v = df[col].dropna().values
        p = stats.wilcoxon(v).pvalue
        print(f'  {name:<22s}  mean={v.mean():+.3f}  median={np.median(v):+.3f}'
              f'  CI=[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]'
              f'  Wilcoxon p={p:.2e}')

    # paired test a > -b  (PEB inequality)
    paired = df[['a_conf', 'b_conf']].dropna()
    pa = stats.wilcoxon(paired['a_conf'] + paired['b_conf']).pvalue  # tests sum != 0
    print(f'\n  Test  a + b  ≠ 0  (sum away from pure-DV null):  '
          f'mean = {(paired.a_conf + paired.b_conf).mean():+.3f}  p = {pa:.2e}')

    # asymmetry index
    df['alpha_conf'] = df.apply(lambda r: asym(r['a_conf'], r['b_conf']),
                                 axis=1)
    print(f'\n  Asymmetry index  α = (a − |b|)/(a + |b|):')
    v = df['alpha_conf'].dropna().values
    p = stats.wilcoxon(v).pvalue
    print(f'    mean = {v.mean():+.3f},  median = {np.median(v):+.3f}, '
          f'Wilcoxon vs 0: p = {p:.2e}')

    print('\n=== conf ~ β_DV·(E⁺−E⁻) + β_Sum·(E⁺+E⁻)   (logistic) ===')
    for col, name in [('beta_DV', 'β_DV  (accuracy axis)'),
                       ('beta_Sum', 'β_Sum (magnitude axis)')]:
        v = df[col].dropna().values
        p = stats.wilcoxon(v).pvalue
        print(f'  {name:<25s}  mean={v.mean():+.3f}  '
              f'p={p:.2e}')

    # same for y
    print('\n=== y_resid ~ a_y·E⁺ + b_y·E⁻  (linear) ===')
    for col, name in [('a_y', 'a_y (chosen pool)'),
                       ('b_y', 'b_y (non-chosen pool)')]:
        v = df[col].dropna().values
        p = stats.wilcoxon(v).pvalue
        print(f'  {name:<25s}  mean={v.mean():+.3f}  '
              f'p={p:.2e}')
    df['alpha_y'] = df.apply(lambda r: asym(r['a_y'], r['b_y']), axis=1)
    v = df['alpha_y'].dropna().values
    if len(v) > 0:
        p = stats.wilcoxon(v).pvalue
        print(f'    asymmetry α_y:  mean={v.mean():+.3f}, '
              f'p vs 0: {p:.2e}')

    df.to_csv(os.path.join(HERE, 'data', 'peb_vs_me.csv'), index=False)

    # ---------- figures ----------
    pdf = PdfPages(os.path.join(FIG, 'peb_vs_me.pdf'))

    # page 1 — (a, b) scatter for conf
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    a = df['a_conf'].values;  b = df['b_conf'].values
    ax.scatter(a, b, s=40, alpha=0.7, color='tab:purple',
                edgecolors='k', linewidths=0.4)
    lim_lo = min(a.min(), b.min()) - 0.1
    lim_hi = max(a.max(), b.max()) + 0.1
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    # reference lines
    rng = np.linspace(lim_lo, lim_hi, 100)
    ax.plot(rng,  rng,  'g--', lw=1, alpha=0.7,
             label='b = a  (pure magnitude)')
    ax.plot(rng, -rng,  'b--', lw=1, alpha=0.7,
             label='b = −a (pure DV / accuracy)')
    ax.plot(rng, np.zeros_like(rng), 'r--', lw=1, alpha=0.7,
             label='b = 0  (pure PEB)')
    ax.set_xlabel('a  =  weight on E⁺  (chosen pool)')
    ax.set_ylabel('b  =  weight on E⁻  (non-chosen pool)')
    ax.set_xlim(lim_lo, lim_hi);  ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect('equal')
    ax.set_title('conf ~ a·E⁺ + b·E⁻   (one dot per session × choice)\n'
                 'Where in (a, b) space do the data sit?')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    # annotate the centroid
    ax.plot(a.mean(), b.mean(), 'k*', ms=18, label='_nolegend_')
    ax.annotate(f'mean ({a.mean():+.2f}, {b.mean():+.2f})',
                 (a.mean(), b.mean()),
                 textcoords='offset points', xytext=(10, -16),
                 fontsize=10)
    fig.tight_layout();  pdf.savefig(fig);  plt.close(fig)

    # page 2 — β_DV vs β_Sum scatter
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    bdv = df['beta_DV'].values;  bsum = df['beta_Sum'].values
    ax.scatter(bdv, bsum, s=40, alpha=0.7, color='tab:orange',
                edgecolors='k', linewidths=0.4)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('β_DV  (weight on E⁺ − E⁻, the accuracy axis)')
    ax.set_ylabel('β_Sum (weight on E⁺ + E⁻, the magnitude axis)')
    ax.plot(bdv.mean(), bsum.mean(), 'k*', ms=18)
    ax.annotate(f'mean ({bdv.mean():+.2f}, {bsum.mean():+.2f})',
                 (bdv.mean(), bsum.mean()),
                 textcoords='offset points', xytext=(10, 8), fontsize=10)
    ax.set_title('conf ~ β_DV·(E⁺−E⁻) + β_Sum·(E⁺+E⁻)\n'
                 'Both non-zero ⇒ confidence uses BOTH the DV axis (accuracy)\n'
                 'AND the orthogonal magnitude axis (gain).')
    fig.tight_layout();  pdf.savefig(fig);  plt.close(fig)

    # page 3 — α distribution
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = df['alpha_conf'].dropna().values
    ax.hist(vals, bins=20, color='tab:purple', alpha=0.75, edgecolor='k')
    ax.axvline(0, color='k', lw=0.6, label='α = 0 (symmetric: either DV or ME)')
    ax.axvline(1, color='tab:red', lw=0.6, ls='--',
                label='α = +1 (pure PEB)')
    ax.axvline(np.mean(vals), color='tab:purple', lw=2.5,
                label=f'mean α = {np.mean(vals):+.2f}')
    ax.set_xlabel('Asymmetry  α = (a − |b|) / (a + |b|)')
    ax.set_ylabel('# (session × choice) pairs')
    ax.set_xlim(-1.1, 1.1)
    ax.set_title('Asymmetry index for  conf ~ a·E⁺ + b·E⁻')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # page 4 — summary text
    fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111); ax.axis('off')
    T = [
        'PEB vs ME — population decomposition (29 sessions, 58 pairs after pref labels)',
        '',
        'conf ~ a · E⁺ + b · E⁻   (logistic, E± z-scored within sess×choice)',
        f'  a (chosen pool)      mean = {df.a_conf.mean():+.3f}   p vs 0 = '
        f'{stats.wilcoxon(df.a_conf.dropna()).pvalue:.2e}',
        f'  b (non-chosen pool)  mean = {df.b_conf.mean():+.3f}   p vs 0 = '
        f'{stats.wilcoxon(df.b_conf.dropna()).pvalue:.2e}',
        f'  Wilcoxon (a+b ≠ 0):  mean = {(df.a_conf + df.b_conf).mean():+.3f}   '
        f'p = {stats.wilcoxon(df.a_conf + df.b_conf).pvalue:.2e}',
        '',
        f'Asymmetry α = (a − |b|)/(a + |b|):  mean = {df.alpha_conf.dropna().mean():+.3f}',
        f'  α ≈ +1 → pure PEB,   α ≈ 0 → symmetric (DV if b<0, ME if b>0)',
        '',
        'Same model in DV / Sum basis:',
        f'  β_DV   (accuracy axis)  = {df.beta_DV.mean():+.3f}   p = '
        f'{stats.wilcoxon(df.beta_DV.dropna()).pvalue:.2e}',
        f'  β_Sum  (magnitude axis) = {df.beta_Sum.mean():+.3f}   p = '
        f'{stats.wilcoxon(df.beta_Sum.dropna()).pvalue:.2e}',
        '',
        'Interpretation rules:',
        '  pure DV / accuracy           a > 0,  b = −a   (sum a+b = 0)',
        '  positive-evidence bias       a > 0,  b ≈ 0    (α near +1)',
        '  magnitude / gain effect      a = b > 0        (α near 0,  with b > 0)',
        '',
        'Reading off the data:',
        f'  Sign of a:   {"+ (positive)" if df.a_conf.mean()>0 else "− (negative)"}',
        f'  Sign of b:   {"+ (positive)" if df.b_conf.mean()>0 else "− (negative)"}',
        f'  Sign of (a+b):  {"+" if (df.a_conf+df.b_conf).mean()>0 else "−"}'
        f'   →  excess magnitude component  (i.e. confidence ≠ pure DV)',
    ]
    ax.text(0.02, 0.96, '\n'.join(T), transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=9)
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print('\nfigure:', os.path.join(FIG, 'peb_vs_me.pdf'))


if __name__ == '__main__':
    main()
