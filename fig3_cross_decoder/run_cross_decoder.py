"""
Choice-conditioned cross-decoder: how big is the AUC gap when we
predict confidence using a decoder trained on accuracy, compared with
a decoder trained on confidence?

Per (session × choice):
  1. CV-fit a linear logistic decoder of  CORRECT     → score y_corr
  2. CV-fit a linear logistic decoder of  CONFIDENCE  → score y_conf
  3. Evaluate AUC of each score against the *confidence* label.
  4. Also: residualise the conf decoder's output against the acc
     decoder's output  →  the part of the conf signal that the acc
     decoder cannot capture, and its AUC for conf.

If the conf-trained decoder has AUC noticeably above the acc-trained
decoder when both are scored against confidence, the gap is the
"non-accuracy" component of confidence.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

# Unified FULL-window spike-count basis (same 50-500ms recount used for
# Fig2/Fig4B/Fig4A/SI-controls) in place of TEST's precomputed
# `stimulus_spikeCounts` -- see fig4_peb_pools/unified_loader.py and
# contrast_fix/build_unified_session_data.py for provenance. This script
# is all-neuron (no preference-pool filtering), so it uses
# load_sessions_unified() directly; sessions 7 and 10 are absent from the
# unified cache (0 significantly right-preferring neurons each -- an
# artifact of how the cache was built for the preference-pool figures,
# not a property of these two sessions' overall neuron counts) so this
# script's N drops from up to 58 to up to 54 session x choice pairs.
sys.path.insert(0, os.path.join(HERE, '..', 'fig4_peb_pools'))
from unified_loader import load_sessions_unified

FIG  = os.path.join(HERE, 'figures');  os.makedirs(FIG, exist_ok=True)


def load_sessions(path=None):
    return load_sessions_unified()


def cv_predict(X, y, seed=0, n_splits=5):
    """Held-out probability predictions for binary y."""
    if np.unique(y).size < 2:  return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def safe_auc(score, y):
    if score is None: return np.nan
    try: return roc_auc_score(y, score)
    except: return np.nan


def fit_full(X, y, C=1.0):
    """Refit on all data (same architecture) to get a single weight vector
    representative of the decoder's readout direction in neuron space."""
    if np.unique(y).size < 2:
        return None
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
    return clf.coef_.ravel()


def cosine_sim(a, b):
    if a is None or b is None: return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float((a @ b) / (na * nb))


def main():
    sess = load_sessions_unified()
    rows = []
    for s in sess:
        for ch, key in [(+1, 'CW'), (-1, 'CCW')]:
            mask = s['choice'] == ch
            if mask.sum() < 80:  continue
            X = s['spk'][mask, :]
            y_conf = s['conf'][mask]
            y_acc  = s['correct'][mask]
            if (min(np.bincount(y_conf, minlength=2)) < 10 or
                min(np.bincount(y_acc , minlength=2)) < 10):
                continue
            # decoders
            p_conf = cv_predict(X, y_conf)         # trained on conf
            p_acc  = cv_predict(X, y_acc)          # trained on accuracy
            if p_conf is None or p_acc is None:  continue

            auc_conf_on_conf = safe_auc(p_conf, y_conf)
            auc_acc_on_conf  = safe_auc(p_acc,  y_conf)
            auc_acc_on_acc   = safe_auc(p_acc,  y_acc)
            auc_conf_on_acc  = safe_auc(p_conf, y_acc)

            # residual of p_conf after removing p_acc (linear regress)
            # logit-transform both then partial out, to keep on a sensible scale
            def logit(p):
                p = np.clip(p, 1e-4, 1 - 1e-4)
                return np.log(p / (1 - p))
            l_conf = logit(p_conf)
            l_acc  = logit(p_acc)
            A = np.column_stack([np.ones_like(l_acc), l_acc])
            beta, *_ = np.linalg.lstsq(A, l_conf, rcond=None)
            resid = l_conf - A @ beta
            auc_resid_on_conf = safe_auc(resid, y_conf)

            # weight vectors (refit on all data) and their cosine similarity
            w_conf = fit_full(X, y_conf)
            w_acc  = fit_full(X, y_acc)
            cos_conf_acc = cosine_sim(w_conf, w_acc)

            rows.append(dict(
                session=s['idx'], choice=key, n=int(mask.sum()),
                auc_conf_on_conf=auc_conf_on_conf,
                auc_acc_on_conf =auc_acc_on_conf,
                auc_acc_on_acc  =auc_acc_on_acc,
                auc_conf_on_acc =auc_conf_on_acc,
                auc_resid_on_conf=auc_resid_on_conf,
                cos_conf_acc=cos_conf_acc))

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'{len(df)} pair rows')
    print('\nMean AUCs (each row is a session × choice pair):')
    for c in ['auc_conf_on_conf', 'auc_acc_on_conf',
              'auc_acc_on_acc',   'auc_conf_on_acc', 'auc_resid_on_conf']:
        v = df[c].dropna().values
        print(f'  {c:<22s} mean={v.mean():.3f}  median={np.median(v):.3f}')

    # paired test
    paired = df[['auc_conf_on_conf', 'auc_acc_on_conf']].dropna()
    diff = paired['auc_conf_on_conf'] - paired['auc_acc_on_conf']
    p = stats.wilcoxon(diff).pvalue
    print(f'\nΔ (conf-trained − acc-trained), both scored on conf:')
    print(f'  mean = {diff.mean():+.4f}, median = {diff.median():+.4f}, '
          f'Wilcoxon p = {p:.2e}, n = {len(diff)}')

    # mirror-image paired test: both decoders scored on ACCURACY
    paired_acc = df[['auc_acc_on_acc', 'auc_conf_on_acc']].dropna()
    diff_acc = paired_acc['auc_acc_on_acc'] - paired_acc['auc_conf_on_acc']
    p_acc = stats.wilcoxon(diff_acc).pvalue
    print(f'\nΔ (acc-trained − conf-trained), both scored on acc:')
    print(f'  mean = {diff_acc.mean():+.4f}, median = {diff_acc.median():+.4f}, '
          f'Wilcoxon p = {p_acc:.2e}, n = {len(diff_acc)}')

    # paired test for residual
    paired2 = df[['auc_acc_on_conf', 'auc_resid_on_conf']].dropna()
    print(f'\nResidual AUC on conf  (after partialling out acc-trained '
          f'output): mean = {paired2.auc_resid_on_conf.mean():.3f}')

    # cosine similarity between the conf-trained and acc-trained weight vectors
    cs = df['cos_conf_acc'].dropna()
    print(f'\ncos(w_conf, w_acc) across {len(cs)} (session x choice) pairs:')
    print(f'  mean={cs.mean():+.3f}  median={cs.median():+.3f}  '
          f'SEM={cs.std(ddof=1)/np.sqrt(len(cs)):.3f}')

    # export per-pair results for downstream plotting (e.g. the MATLAB
    # publication-figure script make_pub_figure.m in this same folder)
    DATAOUT = os.path.join(HERE, 'data');  os.makedirs(DATAOUT, exist_ok=True)
    df.to_csv(os.path.join(DATAOUT, 'cross_decoder_pairs.csv'), index=False)
    try:
        import scipy.io as sio
        mat_dict = {c: df[c].to_numpy() for c in df.columns if c != 'choice'}
        mat_dict['choice_sign'] = np.where(df['choice'].values == 'CW', 1.0, -1.0)
        sio.savemat(os.path.join(DATAOUT, 'cross_decoder_pairs.mat'), mat_dict)
        print('data ->', os.path.join(DATAOUT, 'cross_decoder_pairs.mat'))
    except Exception as e:
        print('(skipping .mat export:', e, ')')

    # save figure
    pdf = PdfPages(os.path.join(FIG, 'cross_decoder.pdf'))

    # page 1: bar comparison
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labs = ['conf-trained\non conf', 'acc-trained\non conf',
            'acc-trained\non acc', 'conf residual\n(orth to acc)\non conf']
    vals = [df.auc_conf_on_conf, df.auc_acc_on_conf,
            df.auc_acc_on_acc, df.auc_resid_on_conf]
    colors = ['tab:red', 'tab:orange', 'tab:gray', 'tab:purple']
    for j, (v, lab, col) in enumerate(zip(vals, labs, colors)):
        v = v.dropna().values
        ax.bar(j, v.mean(), yerr=v.std()/np.sqrt(len(v)),
                color=col, alpha=0.8, capsize=4)
        ax.scatter(np.full(len(v), j) +
                    np.random.uniform(-.12, .12, len(v)),
                    v, color='k', s=14, alpha=0.45)
    ax.axhline(0.5, color='k', lw=0.5)
    ax.set_xticks(range(4));  ax.set_xticklabels(labs)
    ax.set_ylabel('AUC')
    ax.set_title('Choice-conditioned cross-decoder')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # page 2: paired scatter conf-on-conf vs acc-on-conf
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.auc_acc_on_conf, df.auc_conf_on_conf,
                s=35, alpha=0.7, color='tab:red')
    lo = min(df.auc_acc_on_conf.min(), df.auc_conf_on_conf.min()) - 0.02
    hi = max(df.auc_acc_on_conf.max(), df.auc_conf_on_conf.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.6, alpha=0.5)
    ax.set_xlabel('AUC of acc-trained decoder on conf')
    ax.set_ylabel('AUC of conf-trained decoder on conf')
    ax.set_title('Each dot = one (session × choice) pair\n'
                 'Diagonal would mean "conf is fully an accuracy estimate"')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # page 3: histogram of cos(w_conf, w_acc) across sessions
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(cs, bins=20, color='tab:blue', alpha=0.75, edgecolor='k')
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(cs.mean(), color='tab:red', lw=2,
               label=f'mean = {cs.mean():+.2f}')
    ax.axvline(1, color='k', lw=0.3, ls='--', alpha=0.5)
    ax.axvline(-1, color='k', lw=0.3, ls='--', alpha=0.5)
    ax.set_xlabel('cos(w_conf, w_acc)  (same session × choice)')
    ax.set_ylabel('# (session × choice) pairs')
    ax.set_title('Are the conf-trained and acc-trained readout directions\n'
                 'aligned, orthogonal, or opposed?')
    ax.legend(frameon=False)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # page 4: summary text
    fig = plt.figure(figsize=(9, 4.6)); ax = fig.add_subplot(111); ax.axis('off')
    n_sessions = df['session'].nunique()
    T = [
        f'Choice-conditioned cross-decoder ({n_sessions} sessions x 2 choices = {len(df)} pairs)',
        '',
        f'AUC (mean across pairs):',
        f'  conf-trained, scored on conf  = {df.auc_conf_on_conf.mean():.3f}',
        f'  acc-trained,  scored on conf  = {df.auc_acc_on_conf.mean():.3f}',
        f'  acc-trained,  scored on acc   = {df.auc_acc_on_acc.mean():.3f}',
        f'  residual (conf orth to acc),  = {df.auc_resid_on_conf.mean():.3f}',
        f'    scored on conf                ',
        '',
        f'Δ AUC (conf-trained − acc-trained, scored on conf):',
        f'  mean = {diff.mean():+.4f},  Wilcoxon p = {p:.2e}',
        '',
        f'cos(w_conf, w_acc) across {len(cs)} pairs:',
        f'  mean = {cs.mean():+.3f},  median = {cs.median():+.3f}',
        '',
        'Interpretation:',
        '  The two decoders are trained on the SAME neural activity, the SAME ',
        '  trials, with the SAME architecture (linear logistic).  The only ',
        '  difference is the *training label* (conf vs correct).',
        '  Any AUC gap therefore reflects a genuine difference in the ',
        '  population codes for confidence vs correctness — the brain is ',
        '  reading a different linear combination of V1 activity when it ',
        '  reports confidence than would be required to predict correctness.',
        '  The cosine similarity above quantifies this directly: it is the ',
        '  angle between the conf-trained and acc-trained readout weight ',
        '  vectors in neuron space (1 = identical direction, 0 = orthogonal, ',
        '  -1 = opposite).']
    ax.text(0.02, 0.98, '\n'.join(T), transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=9)
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print('\nfigure:', os.path.join(FIG, 'cross_decoder.pdf'))


if __name__ == '__main__':
    main()
