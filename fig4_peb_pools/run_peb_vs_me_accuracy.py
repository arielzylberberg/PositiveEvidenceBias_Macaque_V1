"""
Positive-Evidence Bias  vs  Magnitude Effect  —  ACCURACY version
==================================================================

This is the accuracy analogue of
`../confidence_residual_axis/run_peb_vs_me.py`.

The only substantive change: the dependent variable is **correctness**
(`correct`) instead of confidence.  We ask whether, when you predict
ACCURACY from the chosen / non-chosen pools, the coefficients sit in a
different place in (a, b) space than they did for confidence.

    correct  ~  a · E⁺  +  b · E⁻

where the chosen vs non-chosen pools are defined by the preference
labels in `../neuron_pref/neuron_pref.mat`, swapped per choice (so E⁺
is always the pool that matches the choice that was made).

Diagnostic for the (a, b) pair:

  *  pure DV / accuracy readout      a = +c,  b = −c     (sign-flipped)
  *  POSITIVE-EVIDENCE BIAS          a > 0,  b ≈ 0 or < 0,  |a| > |b|
  *  MAGNITUDE / GAIN EFFECT         a = b > 0           (symmetric, both +)
  *  mixed PEB+ME                    a > b > 0           (both +, but a > b)

Prior expectation: accuracy *is* the decision variable, so the
accuracy coefficients should sit much closer to the pure-DV line
(b = −a) than the confidence coefficients did.  That contrast is the
whole point of this script.

The confidence-residual orthogonal axis ('y_resid') from the original
script is DROPPED here — residualizing accuracy against itself is
degenerate.

We compute one (a, b) pair per (session × choice) and pool across both
choices per session.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_loader import load_sessions_unified  # noqa: E402
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'Boundy-Singer_PNAS2024', 'Structures',
                    'masterNeuralStruct_TEST.mat')
PREF = os.path.join(HERE, '..', 'shared', 'neuron_pref.mat')
FIG  = os.path.join(HERE, 'figures');  os.makedirs(FIG, exist_ok=True)
MIN_POOL_NEURONS = 3  # see run_peb_vs_me.py for rationale
os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)

COL_CONT_IND   = 12 - 1
COL_ORIENT     = 16 - 1
COL_RT         = 28 - 1
COL_CHOICE     = 29 - 1
COL_CONFIDENCE = 30 - 1
COL_RESPONSE   = 31 - 1


def load_sessions(path):
    """Read the v7.3 (HDF5) master struct directly with h5py, one session
    at a time, keeping only the small derived arrays.  This avoids mat73's
    eager full-tree load, which exhausts memory on large files.

    Note on orientation: h5py returns MATLAB matrices transposed relative
    to mat73, so spike counts come in as (neurons, trials) and trialMatrix
    as (cols, trials); we transpose both to (trials, ·) to match the
    original (trials × neurons) / (trials × cols) layout.
    """
    f = h5py.File(path, 'r')
    mb = f['masterNeuralStruct']
    n_sess = mb.shape[0]
    sess = []
    for i in range(n_sess):
        g = f[mb[i, 0]]
        ori_list = np.array(g['dataStructTask']['trial_ori_list']).ravel()
        spk = np.array(g['dataStructTask']['stimulus_spikeCounts']).T  # -> (trials, neurons)
        bh = g['behavior']['trialMatrix']
        if 'trialMatrix_all' in bh.keys():
            tma = np.array(bh['trialMatrix_all']).T                    # -> (trials, cols)
            keep0 = np.isin(tma[:, COL_RESPONSE], [-2, 1]) & ~np.isnan(ori_list)
            tm = tma[keep0, :]; spk = spk[keep0, :]
        else:
            tm = np.array(bh['trialMatrix']).T                         # -> (trials, cols)
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


# ----------------------------------------------------------------------
def per_choice_decomp(s, pref):
    """Return dict per choice with E+/E- and (a, b) coefficients for
    the ACCURACY target."""
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
        y_acc  = s['correct'][mask]
        # need both correct and error trials present
        if min(np.bincount(y_acc, minlength=2)) < 10:
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

        # ---- fit  correct ~ E+ + E-  (logistic) ----
        Xreg = np.column_stack([E_plus, E_minus])
        clf = LogisticRegression(C=10.0, max_iter=500).fit(Xreg, y_acc)
        a_acc, b_acc = clf.coef_.ravel()

        # ---- fit also  correct ~ DV + Sum  for explicit decomposition ----
        DV  = E_plus - E_minus
        Sum = E_plus + E_minus
        clf2 = LogisticRegression(C=10.0, max_iter=500).fit(
            np.column_stack([DV, Sum]), y_acc)
        beta_DV, beta_Sum = clf2.coef_.ravel()

        out[ch] = dict(
            n=int(mask.sum()),
            a_acc=float(a_acc), b_acc=float(b_acc),
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
                **{k: r[k] for k in ('n', 'a_acc', 'b_acc',
                                       'beta_DV', 'beta_Sum')}))
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'{len(df)} (session × choice) rows.')

    # ---------------- summary stats ----------------
    print('\n=== correct ~ a·E⁺ + b·E⁻  (logistic, raw spike z-scores) ===')
    for col, name in [('a_acc', 'a (chosen pool)'),
                       ('b_acc', 'b (non-chosen pool)')]:
        v = df[col].dropna().values
        p = stats.wilcoxon(v).pvalue
        print(f'  {name:<22s}  mean={v.mean():+.3f}  median={np.median(v):+.3f}'
              f'  CI=[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]'
              f'  Wilcoxon p={p:.2e}')

    # paired test a + b  (PEB inequality / departure from pure DV)
    paired = df[['a_acc', 'b_acc']].dropna()
    pa = stats.wilcoxon(paired['a_acc'] + paired['b_acc']).pvalue
    print(f'\n  Test  a + b  ≠ 0  (sum away from pure-DV null):  '
          f'mean = {(paired.a_acc + paired.b_acc).mean():+.3f}  p = {pa:.2e}')

    # asymmetry index
    df['alpha_acc'] = df.apply(lambda r: asym(r['a_acc'], r['b_acc']),
                                 axis=1)
    print(f'\n  Asymmetry index  α = (a − |b|)/(a + |b|):')
    v = df['alpha_acc'].dropna().values
    p = stats.wilcoxon(v).pvalue
    print(f'    mean = {v.mean():+.3f},  median = {np.median(v):+.3f}, '
          f'Wilcoxon vs 0: p = {p:.2e}')

    print('\n=== correct ~ β_DV·(E⁺−E⁻) + β_Sum·(E⁺+E⁻)   (logistic) ===')
    for col, name in [('beta_DV', 'β_DV  (accuracy axis)'),
                       ('beta_Sum', 'β_Sum (magnitude axis)')]:
        v = df[col].dropna().values
        p = stats.wilcoxon(v).pvalue
        print(f'  {name:<25s}  mean={v.mean():+.3f}  '
              f'p={p:.2e}')

    df.to_csv(os.path.join(HERE, 'data', 'peb_vs_me_accuracy.csv'), index=False)

    # ---------- figures ----------
    pdf = PdfPages(os.path.join(FIG, 'peb_vs_me_accuracy.pdf'))

    # page 1 — (a, b) scatter for accuracy
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    a = df['a_acc'].values;  b = df['b_acc'].values
    ax.scatter(a, b, s=40, alpha=0.7, color='tab:purple',
                edgecolors='k', linewidths=0.4)
    lim_lo = min(a.min(), b.min()) - 0.1
    lim_hi = max(a.max(), b.max()) + 0.1
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
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
    ax.set_title('correct ~ a·E⁺ + b·E⁻   (one dot per session × choice)\n'
                 'Where in (a, b) space do the data sit?')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
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
    ax.set_title('correct ~ β_DV·(E⁺−E⁻) + β_Sum·(E⁺+E⁻)\n'
                 'If accuracy is a pure DV readout, β_DV ≫ 0 while\n'
                 'β_Sum (the orthogonal magnitude axis) sits near 0.')
    fig.tight_layout();  pdf.savefig(fig);  plt.close(fig)

    # page 3 — α distribution
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = df['alpha_acc'].dropna().values
    ax.hist(vals, bins=20, color='tab:purple', alpha=0.75, edgecolor='k')
    ax.axvline(0, color='k', lw=0.6, label='α = 0 (symmetric: either DV or ME)')
    ax.axvline(1, color='tab:red', lw=0.6, ls='--',
                label='α = +1 (pure PEB)')
    ax.axvline(np.mean(vals), color='tab:purple', lw=2.5,
                label=f'mean α = {np.mean(vals):+.2f}')
    ax.set_xlabel('Asymmetry  α = (a − |b|) / (a + |b|)')
    ax.set_ylabel('# (session × choice) pairs')
    ax.set_xlim(-1.1, 1.1)
    ax.set_title('Asymmetry index for  correct ~ a·E⁺ + b·E⁻')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # page 4 — summary text
    fig = plt.figure(figsize=(9, 5)); ax = fig.add_subplot(111); ax.axis('off')
    T = [
        f'PEB vs ME — ACCURACY target ({df.session.nunique()} sessions, {len(df)} pairs after pref labels)',
        '',
        'correct ~ a · E⁺ + b · E⁻   (logistic, E± z-scored within sess×choice)',
        f'  a (chosen pool)      mean = {df.a_acc.mean():+.3f}   p vs 0 = '
        f'{stats.wilcoxon(df.a_acc.dropna()).pvalue:.2e}',
        f'  b (non-chosen pool)  mean = {df.b_acc.mean():+.3f}   p vs 0 = '
        f'{stats.wilcoxon(df.b_acc.dropna()).pvalue:.2e}',
        f'  Wilcoxon (a+b ≠ 0):  mean = {(df.a_acc + df.b_acc).mean():+.3f}   '
        f'p = {stats.wilcoxon(df.a_acc + df.b_acc).pvalue:.2e}',
        '',
        f'Asymmetry α = (a − |b|)/(a + |b|):  mean = {df.alpha_acc.dropna().mean():+.3f}',
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
        f'  Sign of a:   {"+ (positive)" if df.a_acc.mean()>0 else "− (negative)"}',
        f'  Sign of b:   {"+ (positive)" if df.b_acc.mean()>0 else "− (negative)"}',
        f'  Sign of (a+b):  {"+" if (df.a_acc+df.b_acc).mean()>0 else "−"}',
        f'  (a+b ≈ 0 and b < 0  →  accuracy ≈ pure DV, as expected)',
    ]
    ax.text(0.02, 0.96, '\n'.join(T), transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=9)
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print('\nfigure:', os.path.join(FIG, 'peb_vs_me_accuracy.pdf'))


if __name__ == '__main__':
    main()
