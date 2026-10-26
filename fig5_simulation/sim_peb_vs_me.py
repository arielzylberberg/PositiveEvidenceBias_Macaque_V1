"""
Does confidence read out of a simulated V1 population show a positive
evidence bias (PEB)?

Reuses the generative model from sim_v1_confidence.py (same encoding, same
decision rule, same readouts used for the choice/accuracy "d,g" map
analysis) and asks a different question of it: fit the same (a,b)
decomposition used on the real data in
analysis_az/peb_vs_me_pub_matlab/compute_ab_marginal.m --

    target ~ a . E+  +  b . E-      (logistic regression)

where E+ is the mean z-scored activity of the chosen-preferring neuron
pool and E- of the non-chosen-preferring pool, z-scored *within each
choice side separately* (exactly as in compute_ab_marginal.m Panel A).
Reference regimes:
    b = -a   ->  confidence/accuracy reads the full decision variable e
                 = E+ - E- symmetrically (optimal / "mean evidence", ME)
    b = 0    ->  confidence depends only on evidence FOR the choice made,
                 completely ignoring disconfirming evidence (pure PEB)
    b = +a   ->  confidence depends on E+ + E- (pure gain/magnitude, the
                 "g axis" of the companion d/g map analysis)

Targets tested (six "sensible" models -- readouts that were built to
reproduce the choice/accuracy map geometry or estimate reliability/
uncertainty in a principled way, not to produce a bias -- plus the choice
anchor):
    - choice     : sanity anchor. Choice is fit on the *unconditioned*
                   right-/left-preferring pool activity (x_R, x_L) rather
                   than choice-conditioned E+/E- (choice can't be regressed
                   on a decomposition that already presupposes it). Choice
                   is a symmetric, full-evidence readout by construction
                   (e = z.w = x_R - x_L up to the classification/z-scoring
                   approximation), so it anchors the "-b/a = 1" end.
    - accuracy      : benchmark -- should sit closer to b=-a than confidence.
    - ContrastBlind : Phi(|e|/sigma_pooled), a single contrast-INDEPENDENT
                      reliability estimate (pooled across both contrast
                      conditions) -- a simpler, less "sensible" heuristic
                      than Bayesian(|e|) below, included to test whether
                      not knowing the trial's reliability condition changes
                      the picture.
    - CASANDRE      : high if |e/sig_hat| > crit, sig_hat a noisy,
                      contrast-conditional estimate of trial reliability
                      with LOG-NORMAL meta-noise (sig_hat = sig_c *
                      exp(meta_unc . N(0,1))), matching the generative model
                      in Boundy-Singer, Ziemba & Goris (2023).
    - Bayesian(|e|) : Phi(|e|/sig_c), contrast-conditional reliability known
                      exactly -- CASANDRE with zero meta-noise.
    - Entropy       : negative Shannon entropy of the FULL posterior over
                      the discrete stimulus x contrast grid -- summarizes
                      the whole shape of the population posterior rather
                      than collapsing it to a sign-marginal probability
                      first (contrast with FullBayes(r) below), so it is
                      not a monotonic reparametrization of FullBayes(r).
    - FullBayes(r)  : P(chosen side correct | full population vector r),
                      marginalising over stimulus and contrast with the
                      true generative model. Conditions on all of r, not
                      on the 1-D collapse |e|.

A 7th, deliberately-biased readout (confidence ~ E+ - 0.3*E-) was used
earlier as a positive control to confirm this method recovers a known
asymmetry (ratio ~0.28 recovered exactly) -- kept here as a one-line sanity
check, not part of the reported figures.

Empirical reference (masterNeuralStruct_FULL.mat, see peb_vs_me_pub_matlab/README.md):
    confidence:  a ~ +1.03, b ~ -0.28   (weight ratio -b/a ~ 0.27 -> PEB-like)
    accuracy:    a ~ +1.21, b ~ -0.74   (weight ratio -b/a ~ 0.61 -> more symmetric)
"""
import csv
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression

import sim_v1_confidence as M

HERE = Path(__file__).resolve().parent          # .../fig5_simulation
ROOT = HERE                                     # kept for relative_to() calls below
SHARED_DIR = HERE.parent / 'shared'              # .../code_for_sharing/shared
FIG_DIR = HERE / 'figures'; FIG_DIR.mkdir(exist_ok=True)
DATA_DIR = HERE / 'data'; DATA_DIR.mkdir(exist_ok=True)

# ---- shared publication style (same palette/fonts as every other main-text
# figure -- see analysis_az/style/pub_style.py; COL_CONF_TGT/COL_ACC_TGT and
# COL_CHOSEN/COL_UNCHOSEN mirror the exact usage in
# comments_followup_conf_acc/make_figure4_composite.py, i.e. Figure 4 /
# fig:peb_pools of main4.tex) ------------------------------------------------
sys.path.insert(0, str(SHARED_DIR))
import pub_style as PS  # noqa: E402
PS.apply_rcparams()

COL_ACC_TGT = PS.CONF_LOW     # cool steel-blue -- accuracy, as a target (matches Fig. 4A)
COL_CONF_TGT = PS.CONF_HIGH   # warm amber-red  -- confidence-family readouts (matches Fig. 4A)
COL_CHOSEN = PS.CHOSEN        # green  -- E+ / chosen pool (matches Fig. 4B-D)
COL_UNCHOSEN = PS.UNCHOSEN    # salmon -- E- / unchosen pool (matches Fig. 4B-D)
REF = "0.35"                  # reference-line gray (matches make_figure4_composite.py)

# Across-pair mean "star" colors -- fully-saturated red/blue with a white
# halo edge, distinct from COL_CONF_TGT/COL_ACC_TGT, so the star reads as a
# clear summary marker rather than just a bigger dot (matches the Fig. 4A
# star treatment in make_figure4_composite.py).
STAR_ACC = "#0a3fff"
STAR_CONF = "#e8000b"
STAR_COLOR = {'accuracy': STAR_ACC, 'ContrastBlind': STAR_CONF,
              'CASANDRE': STAR_CONF, 'Bayesian(|e|)': STAR_CONF,
              'Entropy': STAR_CONF, 'FullBayes(r)': STAR_CONF,
              'PartialPositiveEvidence': STAR_CONF}

SEED = 7
LAMBDA_PEB = 0.3          # discount factor on disconfirming evidence for the one-line positive-control check
MIN_TRIALS_CHOICE = 40    # same thresholds as compute_ab_marginal.m
MIN_CLASS_COUNT   = 5
MIN_POOL_NEURONS  = 2
NBOOT = 1000

MAIN_TARGETS = ['accuracy', 'ContrastBlind', 'CASANDRE', 'Bayesian(|e|)', 'Entropy', 'FullBayes(r)',
                'PartialPositiveEvidence']
BAR_TARGETS  = ['choice'] + MAIN_TARGETS   # choice included as the symmetric-integration anchor

rng = default_rng(SEED)

# =============================================================================
# 1. Simulate + process all sessions (identical model/readouts to run_sim_prototype.py)
# =============================================================================
sessions = []
for j in range(M.N_SESSIONS):
    nu = M.UNITS_PER_SESS[j]
    sess = M.simulate_session(nu, M.TRIALS_PER_SESS, rng)
    out = M.process_session(sess, rng)
    if out is not None:
        out['session'] = j
        sessions.append(out)
print(f"Included sessions: {len(sessions)} / {M.N_SESSIONS}")


def cat(key):
    return np.concatenate([s[key] for s in sessions])


choice_all = cat('choice'); acc_all = cat('acc')
casandre = cat('casandre'); bayes = cat('bayes'); fullbayes = cat('fullbayes')
blind = cat('blind'); entropy = cat('entropy')
# PartialPositiveEvidence: chosen/unchosen pool-weighting readout at a
# fixed, 'partially positive' alpha=0.5 (see sim_v1_confidence.py's
# process_session -- chosen+0.5*unchosen), added to test where partial,
# not just total (alpha=0), discounting of the unchosen pool lands on the
# same -b/a scale as the other five 'sensible' confidence computations.
partial_pos = cat('partial_pos')
print(f"Total trials: {choice_all.size}   pooled accuracy = {np.nanmean(acc_all):.3f}   "
      f"P(right) = {choice_all.mean():.3f}")

# ---- global binarization of each continuous confidence readout to P(high)=0.5 ----
def binarize(c, target=M.TARGET_PHIGH):
    thr = np.nanquantile(c, 1 - target)
    return (c >= thr).astype(float)

readout_bin_pooled = {
    'ContrastBlind': binarize(blind),
    'CASANDRE':      binarize(casandre),
    'Bayesian(|e|)': binarize(bayes),
    'Entropy':       binarize(entropy),
    'FullBayes(r)':  binarize(fullbayes),
    'PartialPositiveEvidence': binarize(partial_pos),
}
sizes = [s['d'].size for s in sessions]
splits = np.cumsum(sizes)[:-1]
readout_bin_per_sess = {k: np.split(v, splits) for k, v in readout_bin_pooled.items()}

# =============================================================================
# 2. Panel-A style per-choice E+/E- (re-z-score within each choice side)
# =============================================================================
def per_choice_E(counts, rpref, lpref, choice_arr):
    """Return, for each choice side, (mask, Eplus, Eminus) using Panel-A
    z-scoring (within that choice's trials only), matching compute_ab_marginal.m."""
    out = {}
    for ch in (1, 0):
        mask = choice_arr == ch
        if mask.sum() < MIN_TRIALS_CHOICE:
            continue
        Xc = counts[mask]
        mu = Xc.mean(0); sd = Xc.std(0)
        ok = sd > 0
        if ok.sum() < 4:
            continue
        Xz = np.full_like(Xc, np.nan)
        Xz[:, ok] = (Xc[:, ok] - mu[ok]) / sd[ok]
        if ch == 1:
            chosen_pool, nonchosen_pool = rpref & ok, lpref & ok
        else:
            chosen_pool, nonchosen_pool = lpref & ok, rpref & ok
        if chosen_pool.sum() < MIN_POOL_NEURONS or nonchosen_pool.sum() < MIN_POOL_NEURONS:
            continue
        Eplus = np.nanmean(Xz[:, chosen_pool], 1)
        Eminus = np.nanmean(Xz[:, nonchosen_pool], 1)
        out[ch] = dict(mask=mask, Eplus=Eplus, Eminus=Eminus)
    return out


per_sess_E = [per_choice_E(s['counts'], s['rpref'], s['lpref'], s['choice']) for s in sessions]

# =============================================================================
# 3. Fit target ~ a.E+ + b.E- per (session, choice), for the 6 main targets;
#    fit choice ~ a.x_R + b.x_L per session (unconditioned -- see docstring).
# =============================================================================
rows = []  # each: dict(target, session, choice, n, a, b)

for i, s in enumerate(sessions):
    E = per_sess_E[i]
    for ch, blk in E.items():
        mask, Eplus, Eminus = blk['mask'], blk['Eplus'], blk['Eminus']
        Xr_full = np.column_stack([Eplus, Eminus])

        def fit_target(name, y_full):
            y = y_full[mask]
            ok = np.isfinite(y)
            if ok.sum() < MIN_TRIALS_CHOICE - 10:
                return
            yv = y[ok]
            if min((yv == 0).sum(), (yv == 1).sum()) < MIN_CLASS_COUNT:
                return
            clf = LogisticRegression(C=10, solver='lbfgs', max_iter=2000)
            clf.fit(Xr_full[ok], yv)
            a_, b_ = clf.coef_[0]
            rows.append(dict(target=name, session=s['session'], choice=ch,
                              n=int(ok.sum()), a=a_, b=b_))

        fit_target('accuracy', s['acc'])
        for k in ['ContrastBlind', 'CASANDRE', 'Bayesian(|e|)', 'Entropy', 'FullBayes(r)',
                  'PartialPositiveEvidence']:
            fit_target(k, readout_bin_per_sess[k][i])

    # ---- choice: unconditioned fit on (x_R, x_L), one row per session ----
    d_s, g_s = s['d'], s['g']
    x_R = (g_s + d_s) / 2.0
    x_L = (g_s - d_s) / 2.0
    y_ch = s['choice'].astype(float)
    if min((y_ch == 0).sum(), (y_ch == 1).sum()) >= MIN_CLASS_COUNT:
        clf = LogisticRegression(C=10, solver='lbfgs', max_iter=2000)
        clf.fit(np.column_stack([x_R, x_L]), y_ch)
        a_, b_ = clf.coef_[0]
        rows.append(dict(target='choice', session=s['session'], choice=np.nan,
                          n=y_ch.size, a=a_, b=b_))

print(f"\nFitted {len(rows)} (target x session[x choice]) (a,b) pairs.")

# ---- one-line positive-control sanity check (not part of the reported figures) ----
peb_raw_all, peb_bins = [], []
for E in per_sess_E:
    for ch, blk in E.items():
        peb_raw_all.append(blk['Eplus'] - LAMBDA_PEB * blk['Eminus'])
peb_raw_all = np.concatenate(peb_raw_all)
peb_thr = np.nanquantile(peb_raw_all, 1 - M.TARGET_PHIGH)
check_a, check_b = [], []
for E in per_sess_E:
    for ch, blk in E.items():
        Xr = np.column_stack([blk['Eplus'], blk['Eminus']])
        yv = (blk['Eplus'] - LAMBDA_PEB * blk['Eminus'] >= peb_thr).astype(float)
        if min((yv == 0).sum(), (yv == 1).sum()) < MIN_CLASS_COUNT:
            continue
        clf = LogisticRegression(C=10, solver='lbfgs', max_iter=2000).fit(Xr, yv)
        check_a.append(clf.coef_[0][0]); check_b.append(clf.coef_[0][1])
ratio_check = -np.mean(check_b) / np.mean(check_a)
print(f"Positive-control check: readout built as E+ - {LAMBDA_PEB}*E- recovers "
      f"-b/a = {ratio_check:+.2f} (should be close to {LAMBDA_PEB}) -- confirms the "
      f"method detects a known asymmetry when one is actually present.\n")

# =============================================================================
# 4. Summarize: mean (a,b), weight ratio -b/a, session-level bootstrap CI
# =============================================================================
def rows_for(target):
    return [r for r in rows if r['target'] == target]

def weight_ratio_summary(target):
    sub = rows_for(target)
    a = np.array([r['a'] for r in sub]); b = np.array([r['b'] for r in sub])
    sess_ids = np.array([r['session'] for r in sub])
    uniq = np.unique(sess_ids)
    ratio_obs = -np.mean(b) / np.mean(a)
    boots = np.zeros(NBOOT)
    bootrng = default_rng(3)
    for k in range(NBOOT):
        pick = bootrng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([np.where(sess_ids == p)[0] for p in pick])
        boots[k] = -np.mean(b[idx]) / np.mean(a[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(target=target, n_pairs=len(sub), n_sess=uniq.size,
                mean_a=np.mean(a), mean_b=np.mean(b), ratio=ratio_obs, ci=(lo, hi))


print(f"{'target':16s} | {'pairs':>5s} {'sess':>4s} | {'mean a':>7s} {'mean b':>7s} | "
      f"{'-b/a [95% CI]':>24s}   (0=pure PEB, 1=optimal/ME, -1=pure gain)")
summary = {}
for t in BAR_TARGETS:
    r = weight_ratio_summary(t)
    summary[t] = r
    print(f"{t:16s} | {r['n_pairs']:5d} {r['n_sess']:4d} | {r['mean_a']:+7.2f} {r['mean_b']:+7.2f} | "
          f"{r['ratio']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]")

print("\nEmpirical reference (masterNeuralStruct_FULL.mat, peb_vs_me_pub_matlab):")
print("  confidence : a~+1.03 b~-0.28  -> -b/a ~ 0.27  (PEB-like)")
print("  accuracy   : a~+1.21 b~-0.74  -> -b/a ~ 0.61  (more symmetric)")

print("""
Caveat: absolute (a,b) magnitudes here are much larger than the empirical
~1 scale because these synthetic readouts are near-deterministic functions
of the same-trial population activity (no independent report/criterion
noise), which pushes logistic regression toward quasi-complete separation;
-b/a is stable from C=1 to C=1e5 (checked separately) and is the number to
read, not the raw a,b. CASANDRE / Bayesian(|e|) land ABOVE the b=-a line
(-b/a > 1, nominally over-weighting disconfirming evidence): this reflects
|e| being a NONLINEAR function of e = z.w with the fixed slope-based
weights w, whereas E+/E- are EQUAL-weighted averages over the
t-test-classified pools, so a linear (a,b) fit of a nonlinear readout onto
(E+,E-) need not land on either reference line. It is not a sign of an
"anti-PEB" computation, and the same mismatch would apply to any
|e|-collapsing readout measured this way on real data too.
""")

# =============================================================================
# 5. Normalised weight summary (unit-length (a,b) -> comparable bar heights;
#    see caveat above on why raw a,b magnitudes are not comparable across
#    readouts). Computed before the figure since panel B needs it.
# =============================================================================
def normalized_weight_summary(target):
    sub = rows_for(target)
    a = np.array([r['a'] for r in sub]); b = np.array([r['b'] for r in sub])
    norm = np.sqrt(a**2 + b**2)
    na, nb = a / norm, b / norm
    sess_ids = np.array([r['session'] for r in sub])
    uniq = np.unique(sess_ids)
    obs = (na.mean(), nb.mean())
    boots = np.zeros((NBOOT, 2))
    bootrng = default_rng(5)
    for k in range(NBOOT):
        pick = bootrng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([np.where(sess_ids == p)[0] for p in pick])
        boots[k] = na[idx].mean(), nb[idx].mean()
    ci = np.percentile(boots, [2.5, 97.5], axis=0)   # shape (2,2): rows=lo/hi, cols=na/nb
    return dict(mean_na=obs[0], mean_nb=obs[1],
                na_ci=(ci[0, 0], ci[1, 0]), nb_ci=(ci[0, 1], ci[1, 1]))


norm_summary = {t: normalized_weight_summary(t) for t in BAR_TARGETS}

# =============================================================================
# 6. Combined publication figure: Panel A = 6 (a,b) scatters (2 rows x 3),
#    Panel B = bar chart of unit-normalised E+/E- weights. Styled to match
#    fig:peb_pools (Figure 4) of main4.tex: same font/size (pub_style),
#    same reference-line convention (solid b=0 / dashed b=-a / dotted b=+a,
#    color "0.35"), same two-layer black+white-edged star for the mean, same
#    green/salmon chosen/unchosen palette, same spines-off + panel-letter style.
# =============================================================================
TARGET_COLOR = {'accuracy': COL_ACC_TGT, 'ContrastBlind': COL_CONF_TGT,
                'CASANDRE': COL_CONF_TGT, 'Bayesian(|e|)': COL_CONF_TGT,
                'Entropy': COL_CONF_TGT, 'FullBayes(r)': COL_CONF_TGT,
                'PartialPositiveEvidence': COL_CONF_TGT}
TARGET_LABEL = {'accuracy': 'Accuracy', 'ContrastBlind': 'ContrastBlind',
                'CASANDRE': 'CASANDRE', 'Bayesian(|e|)': r'Bayesian($|e|$)',
                'Entropy': 'Entropy', 'FullBayes(r)': r'FullBayes($r$)',
                'PartialPositiveEvidence': 'Partial Pos. Evidence\n' + r'($\alpha=0.5$)'}
BAR_LABEL = {'choice': 'Choice\n(anchor)', 'accuracy': 'Accuracy',
             'ContrastBlind': 'Contrast\nBlind', 'CASANDRE': 'CASANDRE',
             'Bayesian(|e|)': 'Bayesian\n' + r'($|e|$)', 'Entropy': 'Entropy',
             'FullBayes(r)': 'FullBayes\n' + r'($r$)',
             'PartialPositiveEvidence': 'Partial Pos.\n' + r'Ev. ($\alpha=0.5$)'}

fig = plt.figure(figsize=(14.6, 10.8))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.8],
                      left=0.06, right=0.99, bottom=0.065, top=0.93,
                      wspace=0.22, hspace=0.95)
axes_top = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(len(MAIN_TARGETS))]
ax_bar = fig.add_subplot(gs[2, :])

for i, (ax, t) in enumerate(zip(axes_top, MAIN_TARGETS)):
    sub = rows_for(t)
    a = np.array([r['a'] for r in sub]); b = np.array([r['b'] for r in sub])
    r = summary[t]
    color = TARGET_COLOR[t]

    lim = max(1.0, np.nanpercentile(np.abs(np.concatenate([a, b])), 95) * 1.15)
    rng_ = np.linspace(-lim, lim, 200)
    ln_peb = ax.axhline(0, color=REF, lw=1.2, ls="-", zorder=1)
    ln_dv, = ax.plot(rng_, -rng_, color=REF, lw=1.2, ls="--", zorder=1)
    ln_gain, = ax.plot(rng_, rng_, color=REF, lw=1.2, ls=":", zorder=1)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)

    ax.scatter(a, b, s=18, facecolor=color, edgecolor="0.15", linewidth=0.4,
               alpha=0.75, zorder=3)
    # vivid, white-edged star (same treatment as Fig. 4A, make_figure4_
    # composite.py) -- a clearer summary marker than the dot cloud's own hue.
    ax.scatter([r['mean_a']], [r['mean_b']], marker="*", s=380,
               facecolor=STAR_COLOR[t], edgecolor="white", linewidth=1.3,
               zorder=5)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=PS.FS_TICK)
    ax.set_xlabel(r"$a$ = weight on $E^{+}$", fontsize=PS.FS_LBL - 1)
    if i % 3 == 0:
        ax.set_ylabel(r"$b$ = weight on $E^{-}$", fontsize=PS.FS_LBL - 1)
    ax.text(0.05, 1.24, TARGET_LABEL[t], transform=ax.transAxes, clip_on=False,
            fontsize=PS.FS_LBL, fontweight="bold", color=color, va="bottom", ha="left")
    ax.text(0.05, 1.06, rf"$-b/a={r['ratio']:+.2f}$", transform=ax.transAxes,
            clip_on=False, fontsize=PS.FS_TICK, va="bottom", ha="left")

# shared reference-line legend -- the 7 targets leave gs[1, 3] (bottom-right
# of the Panel A grid) empty, so the legend lives there instead of floating
# over the bottom margin between Panel A and Panel B.
ax_legend = fig.add_subplot(gs[1, 3])
ax_legend.axis("off")
ax_legend.legend(
    [ln_peb, ln_dv, ln_gain],
    [r"$b=0$ (pure PEB)", r"$b=-a$ (DV read-out)", r"$b=+a$ (pure gain)"],
    loc="center", ncol=1, frameon=False,
    fontsize=PS.FS_LEG, handletextpad=0.6, labelspacing=1.1, handlelength=1.8)

# ---- Panel B: bar chart of unit-normalised weight on E+ (chosen) vs. E- (unchosen)
x = np.arange(len(BAR_TARGETS))
width = 0.34
na_vals = [norm_summary[t]['mean_na'] for t in BAR_TARGETS]
nb_vals = [norm_summary[t]['mean_nb'] for t in BAR_TARGETS]
na_err = [[norm_summary[t]['mean_na'] - norm_summary[t]['na_ci'][0] for t in BAR_TARGETS],
          [norm_summary[t]['na_ci'][1] - norm_summary[t]['mean_na'] for t in BAR_TARGETS]]
nb_err = [[norm_summary[t]['mean_nb'] - norm_summary[t]['nb_ci'][0] for t in BAR_TARGETS],
          [norm_summary[t]['nb_ci'][1] - norm_summary[t]['mean_nb'] for t in BAR_TARGETS]]

ax_bar.bar(x - width/2, na_vals, width, yerr=na_err, capsize=4, ecolor="0.15",
           facecolor=COL_CHOSEN, edgecolor=COL_CHOSEN, linewidth=1.2, alpha=0.9,
           label=r"weight on $E^{+}$ (chosen pool)")
ax_bar.bar(x + width/2, nb_vals, width, yerr=nb_err, capsize=4, ecolor="0.15",
           facecolor=COL_UNCHOSEN, edgecolor=COL_UNCHOSEN, linewidth=1.2, alpha=0.9,
           label=r"weight on $E^{-}$ (unchosen pool)")
ax_bar.axhline(0, color="0.15", lw=0.8, zorder=0)
ln_sym_hi = ax_bar.axhline(1/np.sqrt(2), color=REF, lw=1.1, ls="--", zorder=0)
ax_bar.axhline(-1/np.sqrt(2), color=REF, lw=1.1, ls="--", zorder=0)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([BAR_LABEL[t] for t in BAR_TARGETS], fontsize=PS.FS_TICK + 1)
ax_bar.tick_params(axis="x", length=0)
ax_bar.tick_params(axis="y", labelsize=PS.FS_TICK)
ax_bar.set_ylabel("Unit-normalised weight\n" + r"$(a,b)\,/\,\|(a,b)\|$", fontsize=PS.FS_LBL)
ax_bar.set_ylim(-1.05, 1.05)
ax_bar.spines[["top", "right"]].set_visible(False)
bar_handles, bar_labels = ax_bar.get_legend_handles_labels()
bar_handles.append(ln_sym_hi)
bar_labels.append(r"symmetric integration ($\pm1/\sqrt{2}$)")
ax_bar.legend(bar_handles, bar_labels, fontsize=PS.FS_LEG, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              handlelength=1.4, columnspacing=1.2)

fig.text(0.012, 0.975, "A", fontsize=PS.FS_PANEL, fontweight="bold", va="top", ha="left")
fig.text(0.012, 0.300, "B", fontsize=PS.FS_PANEL, fontweight="bold", va="top", ha="left")

fig_path = FIG_DIR / 'sim_peb_vs_me_composite.pdf'
fig.savefig(fig_path)
fig.savefig(str(fig_path).replace('.pdf', '.png'), dpi=170)
print(f"Saved {fig_path.relative_to(ROOT)} (+ .png)")

# =============================================================================
# 7. Save raw (a,b) pairs + summary table for reuse without re-running
# =============================================================================
pairs_path = DATA_DIR / 'sim_peb_vs_me_pairs.csv'
with open(pairs_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['target', 'session', 'choice', 'n', 'a', 'b'])
    w.writeheader()
    w.writerows(rows)

summary_path = DATA_DIR / 'sim_peb_vs_me_summary.csv'
with open(summary_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['target', 'n_pairs', 'n_sess', 'mean_a', 'mean_b', 'ratio_neg_b_over_a', 'ci_lo', 'ci_hi',
                'mean_norm_a', 'mean_norm_b'])
    for t in BAR_TARGETS:
        r = summary[t]; nr = norm_summary[t]
        w.writerow([t, r['n_pairs'], r['n_sess'], r['mean_a'], r['mean_b'],
                    r['ratio'], r['ci'][0], r['ci'][1], nr['mean_na'], nr['mean_nb']])
print(f"Saved {pairs_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}")
