"""
Simulation companion to analysis_az/empirical_choice_conf_map.

Generative V1 model (orientation tuning + contrast-set gain) -> linear
CW/CCW readout -> six confidence readouts. Each trial is projected into the
SAME (choice-axis d, gain-axis g) neural space used by the empirical analysis,
and we ask which confidence definition reproduces the empirical geometry
(confidence & accuracy load on g, choice loads on d).

Two changes from an earlier version of this simulation (both requested as
direct replacements of the corresponding assumptions, not as additional
robustness checks -- see PAPER_CRITIQUE_V1_ASSUMPTIONS.md for the full
rationale):

(1) Contrast no longer stretches the stimulus coordinate. The previous
    encoding multiplied the stimulus angle by a contrast-dependent
    sensitivity factor (SENS_RATIO) before it entered each neuron's tuning
    curve, i.e. tuning_curve(pref, s*sens): for a neuron with preferred
    orientation phi, this shifts its peak (in physical orientation units)
    to phi/sens, which changes tuning-curve GEOMETRY (effective preferred
    orientation and width), not just response amplitude, and is not how
    contrast is generally understood to act on V1 orientation tuning
    (classically closer to contrast-invariant tuning width, with contrast
    instead shifting response gain / the contrast-response function).
    Contrast now enters ONLY through GAIN_BY_CON, a multiplicative response
    AMPLITUDE gain, leaving each neuron's tuning curve shape (and hence its
    effective preferred orientation) identical across contrast conditions.

(2) The simulated observer's population is no longer the same population
    the "recorded" analysis sees. Previously, the same, small (16-75 unit)
    population both generated choice/accuracy/confidence AND was the
    population the confidence read-outs and the (a,b) pool-weighting
    regression were computed from -- unlike a real experiment, where the
    animal's decision draws on a V1 population far larger than what a
    Utah-array-scale recording samples. Each session now simulates a much
    larger population (N_LARGE units) that determines choice, accuracy, and
    all six confidence read-outs (i.e. the animal's actual, full-population
    decision and confidence computation); the (a,b) pool-weighting
    regression -- the analysis actually compared to the real recorded data
    -- is instead computed from a randomly drawn RECORDED subsample of that
    same population, of the actual per-session size (UNITS_PER_SESS, from
    neuron_pref.mat), exactly mirroring how the real analysis only ever
    sees a small sample of the recorded animal's true sensory population.
    This tests whether the observed chosen/unchosen asymmetry pattern could
    be an artifact of analyzing an incomplete sample of a larger decision
    population, rather than a property of the confidence computation itself.

Readouts (all computed on the FULL, large simulated population):
 (i)   CASANDRE     : high if |e / sigma_hat| > crit, sigma_hat = noisy reliability estimate,
                      sigma_hat = sigma_c * exp(meta_unc * N(0,1)) (log-normal meta-noise,
                      matching the generative model in Boundy-Singer, Ziemba & Goris 2023)
 (ii)  Bayesian     : high if Phi(|e|/sigma_c) > crit  (optimal, reliability known,
                      contrast-conditional sigma_c; the meta_unc=0 limit of CASANDRE)
 (iii) SelfConsist  : split-population sign agreement (trial-level neural reliability)
 (iv)  MapAccuracy  : confidence(trial) = P(correct | d,g) read from smoothed map (ceiling ref)
 (vi)  ContrastBlind: like (ii) but with a single, contrast-independent sigma (the observer
                      does not know / use the trial's reliability condition) -- a simpler,
                      less "sensible" heuristic than (ii)
 (vii) Entropy       : negative entropy of the FULL posterior over the stimulus x contrast
                      grid (the same posterior FullBayes(r) collapses to a sign-marginal
                      probability); a genuinely distributional confidence, not a monotonic
                      reparametrization of (v)

Author: prototype for Ariel / Boundy-Singer reanalysis.
"""
import numpy as np
from numpy.random import default_rng

# ----------------------------------------------------------------------------
# Data-matched parameters (from masterBehaviorStruct crossBlock + neuron_pref)
# ----------------------------------------------------------------------------
RNG = default_rng(7)
N_SESSIONS      = 29
# actual per-session RECORDED unit counts (neuron_pref.mat), reused for
# realism -- this is now the size of the subsample drawn from each
# session's large simulated population for the (a,b) regression, not the
# size of the population that generates choice/accuracy/confidence.
UNITS_PER_SESS  = [30,21,30,17,26,41,75,30,32,21,19,34,19,18,17,23,29,27,40,
                   32,30,44,17,37,41,16,41,35,17]
TRIALS_PER_SESS = 750          # behavioural numTrials ~400-750
WIN_S           = 0.45         # 50-500 ms spike-count window

# Size of the full simulated population that actually determines choice,
# accuracy, and confidence on each trial (the "true" sensory population);
# UNITS_PER_SESS is then a random subsample of this, standing in for what a
# real recording actually captures. ~10-30x the largest recorded session.
N_LARGE         = 500

# contrast (reliability) levels + contrast-dependent response gain
CONTRASTS       = np.array([0.6, 1.2])     # low / high reliability (data ~0.5-0.76 / ~1.2)
GAIN_BY_CON     = np.array([0.6, 1.2])     # multiplicative response-AMPLITUDE gain per contrast
# (Contrast previously also stretched the stimulus coordinate via a
# SENS_RATIO factor inside the tuning curve; removed -- see module
# docstring item (1). Contrast now acts only through GAIN_BY_CON.)

# CASANDRE readout params (data medians)
META_UNC        = 0.7          # meta-uncertainty (noise on reliability estimate)
CONF_CRIT       = 1.0          # criterion on normalized DV (tuned to base rate below)

# orientation stimulus set (deg), signed CW(+)/CCW(-), denser near 0 (~ data)
ORI_LEVELS      = np.array([0.75,1.5,3,4.5,6,9,15])
ORI_SET         = np.concatenate([-ORI_LEVELS, [0.0], ORI_LEVELS])

# encoding: von Mises orientation tuning (period 180 deg)
KAPPA           = 2.0          # tuning width
BASELINE_HZ     = 5.0          # baseline rate
PEAK_HZ         = 25.0         # peak driven rate at contrast gain=1

# map params (identical to empirical pipeline)
GRID_BINS       = 20
PAD             = 0.3
SMOOTH_SIGMA    = 1.5
MIN_SM_TRIALS   = 2
MIN_GRP         = 3
TARGET_PHIGH    = 0.5          # set each readout's criterion to this base rate


# ----------------------------------------------------------------------------
# Encoding model
# ----------------------------------------------------------------------------
def tuning_curve(pref_deg, s_deg):
    """von Mises orientation tuning, normalized 0..1, period 180 deg."""
    x = np.deg2rad(2.0 * (s_deg[None, :] - pref_deg[:, None]))  # (neuron, trial)
    v = np.exp(KAPPA * (np.cos(x) - 1.0))
    return v

def tuning_slope(pref_deg, s0=0.0):
    """d tuning / d s at reference s0 (deg) -> readout weight direction."""
    h = 1e-3
    return (tuning_curve(pref_deg, np.array([s0+h])) -
            tuning_curve(pref_deg, np.array([s0-h]))).ravel() / (2*h)


def simulate_session(n_recorded, n_trials, rng):
    """Simulate a large (N_LARGE-unit) population that determines choice,
    accuracy and confidence, plus a random RECORDED subsample of size
    n_recorded (the real, per-session unit count) drawn from it -- see
    module docstring item (2). Returns raw counts for the FULL population
    (trial x N_LARGE), the generative readout weights, and the recorded
    subsample's indices; process_session does the rest."""
    n_large = N_LARGE
    pref = rng.uniform(-90, 90, n_large)                        # preferred oris
    w    = tuning_slope(pref)                                    # readout weights
    # per-neuron peak drive heterogeneity
    amp  = PEAK_HZ * rng.uniform(0.5, 1.5, n_large)
    base = BASELINE_HZ * rng.uniform(0.5, 1.5, n_large)

    s   = rng.choice(ORI_SET, n_trials)                        # signed orientation
    ci  = rng.integers(0, len(CONTRASTS), n_trials)            # contrast index
    cg  = GAIN_BY_CON[ci]                                       # response-amplitude gain per trial

    tc  = tuning_curve(pref, s)                                # (unit, trial); contrast enters only via cg
    rate = base[:, None] + cg[None, :] * amp[:, None] * tc     # Hz
    mean_count = rate * WIN_S
    counts = rng.poisson(np.maximum(mean_count, 1e-6)).T.astype(float)  # (trial, n_large)

    rec_idx = rng.choice(n_large, n_recorded, replace=False)
    rec_idx.sort()

    return dict(counts=counts, s=s, ci=ci, w=w, pref=pref, amp=amp, base=base,
                rec_idx=rec_idx, n_large=n_large, n_recorded=n_recorded)


def zscore_cols(X):
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
    sd[sd == 0] = np.nan
    return (X - mu) / sd


# ----------------------------------------------------------------------------
# Decision + confidence readouts (per session)
# ----------------------------------------------------------------------------
from scipy.stats import ttest_ind, norm

def fdr_bh(p, alpha=0.05):
    p = np.asarray(p); n = p.size
    order = np.argsort(p); ranked = p[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    sig = np.zeros(n, bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        sig[order[:kmax+1]] = True
    return sig


def conditional_sigma(e_sub, s_sub):
    """Trial-noise SD of e, isolated from stimulus-driven signal variance.

    np.std(e_sub) alone (the original definition of sigma_c/sigma_pooled)
    pools e across every |orientation| magnitude present in e_sub, so it
    reflects stimulus-driven variation in the mean of e (different
    orientations drive different mean |e|) as well as genuine trial-to-trial
    noise -- overstating the noise term that Phi(|e|/sigma) needs to behave
    as a calibrated posterior. This computes the SD of e WITHIN each
    distinct SIGNED stimulus value present in s_sub (so +s and -s trials are
    never pooled together), then trial-count-weights the average across
    those signed values, so stimulus-driven mean shifts are removed and only
    the residual (noise) spread remains. Verified by direct simulation (see
    check_sigma_c.py / SIGMA_C_CHECK_RESULTS.md): the original definition
    overstated the true per-stimulus noise SD by ~1.5x (low contrast) to
    ~2.2x (high contrast) and left Phi(|e|/sigma_c) measurably underconfident
    throughout the reported range.
    """
    uniq_s = np.unique(s_sub)
    stds, weights = [], []
    for sv in uniq_s:
        m = s_sub == sv
        n = m.sum()
        if n > 1:
            stds.append(np.std(e_sub[m]))
            weights.append(n)
    if not weights:
        return np.std(e_sub)
    return float(np.average(stds, weights=weights))


def _classify_pools(z, choice):
    """Two-sample t-test classification of right-/left-preferring units
    from z-scored counts (trial x unit) and a binary choice vector,
    FDR-corrected. Shared by the full-population and recorded-subsample
    classifications below."""
    nun = z.shape[1]
    tval = np.zeros(nun); pval = np.ones(nun)
    R = choice == 1; L = choice == 0
    if R.sum() > 2 and L.sum() > 2:
        for i in range(nun):
            t, p = ttest_ind(z[R, i], z[L, i], equal_var=False, nan_policy='omit')
            tval[i] = t; pval[i] = p
    sig = fdr_bh(pval)
    rpref = sig & (tval > 0)
    lpref = sig & (tval < 0)
    return rpref, lpref


def process_session(sess, rng, n_boot=40):
    counts = sess['counts']; s = sess['s']; ci = sess['ci']; w = sess['w']
    pref = sess['pref']; amp = sess['amp']; base = sess['base']
    rec_idx = sess['rec_idx']
    ntr, n_large = counts.shape

    z = zscore_cols(counts)
    good_u = ~np.all(np.isnan(z), 0)
    if not good_u.all():
        # degenerate (zero-variance) units are vanishingly rare at these
        # rates/trial counts, but guard anyway; drop them from the full
        # population consistently everywhere they are indexed below.
        keep = np.where(good_u)[0]
        counts = counts[:, keep]; z = z[:, keep]; w = w[keep]
        pref = pref[keep]; amp = amp[keep]; base = base[keep]
        remap = -np.ones(n_large, dtype=int); remap[keep] = np.arange(keep.size)
        rec_idx = remap[rec_idx]; rec_idx = rec_idx[rec_idx >= 0]
        n_large = z.shape[1]

    # --- decision variable + confidence, from the FULL (large) population:
    # this is the animal's actual decision process, which by hypothesis
    # draws on far more units than any recording samples (module docstring
    # item (2)) ---
    e = z @ w                              # signed evidence
    choice = (e > 0).astype(int)           # 1 = "right"(CW), 0 = "left"(CCW)
    acc = np.full(ntr, np.nan)             # correct if sign(e)==sign(s); s==0 -> undefined
    nz = s != 0
    acc[nz] = (np.sign(e[nz]) == np.sign(s[nz])).astype(float)

    # --- reliability per contrast: trial-noise SD of e, isolated from
    # stimulus-driven signal variance (see conditional_sigma docstring) ---
    sigma_c = np.array([conditional_sigma(e[ci == k], s[ci == k])
                        for k in range(len(CONTRASTS))])
    sig_tr  = sigma_c[ci]

    # (i) CASANDRE: noisy reliability estimate. Meta-noise is LOG-NORMAL
    # (geometric mean sigma_c, log-SD META_UNC), matching the generative
    # model in Boundy-Singer, Ziemba & Goris (2023) -- sigma cannot be
    # negative, so the noise is multiplicative in log-space rather than the
    # additive-then-folded form used in an earlier version of this script.
    sigma_hat = sig_tr * np.exp(META_UNC * rng.standard_normal(ntr))
    casandre_c = np.abs(e) / (sigma_hat + 1e-9)

    # (ii) Bayesian posterior P(correct|e), reliability known and
    # contrast-conditional (the meta_unc -> 0 limit of CASANDRE)
    bayes_c = norm.cdf(np.abs(e) / (sig_tr + 1e-9))

    # (vi) Contrast-blind confidence: same computation as (ii) but with a
    # single, contrast-INDEPENDENT reliability estimate (pooled SD of e
    # across both contrast conditions) -- a simpler heuristic in which the
    # observer's confidence does not use knowledge of which reliability
    # condition applied on that trial.
    sigma_pooled = conditional_sigma(e, s)
    blind_c = norm.cdf(np.abs(e) / (sigma_pooled + 1e-9))

    # (iii) self-consistency: split-population sign agreement / SNR, on the
    # full population.
    half = max(3, n_large // 2)
    eb = np.zeros((n_boot, ntr))
    for b in range(n_boot):
        idx = rng.choice(n_large, half, replace=False)
        eb[b] = z[:, idx] @ w[idx]
    selfcon_c = np.abs(eb.mean(0)) / (eb.std(0) + 1e-9)

    # (v) FULL Bayesian: P(chosen choice correct | full population vector r),
    #     marginalizing over the discrete stimulus set and both contrasts with
    #     the true generative model (Poisson). Conditions on r (ALL n_large
    #     units), NOT on the 1-D collapse |e|, and not on the recorded
    #     subsample -- this is the animal's own full-information posterior.
    K = ORI_SET.size; J = len(CONTRASTS)
    logM = np.zeros((n_large, K*J)); sumM = np.zeros(K*J); sign_grid = np.zeros(K*J)
    col = 0
    for j in range(J):
        for k in range(K):
            tc_k = tuning_curve(pref, np.array([ORI_SET[k]])).ravel()  # (n_large,); no sens stretch
            m = np.maximum(WIN_S * (base + GAIN_BY_CON[j]*amp*tc_k), 1e-6)
            logM[:, col] = np.log(m); sumM[col] = m.sum(); sign_grid[col] = np.sign(ORI_SET[k])
            col += 1
    loglik = counts @ logM - sumM[None, :]          # (ntr, K*J), uniform prior
    loglik -= loglik.max(1, keepdims=True)
    post = np.exp(loglik); post /= post.sum(1, keepdims=True)
    p_pos = post[:, sign_grid > 0].sum(1)           # P(theta>0 | r)
    p_neg = post[:, sign_grid < 0].sum(1)           # P(theta<0 | r)
    fullbayes_c = np.where(e > 0, p_pos, p_neg)      # P(chosen sign correct | r)

    # (vii) Entropy: negative Shannon entropy of the FULL posterior over the
    # stimulus x contrast grid (same `post` as FullBayes(r), but summarizing
    # its whole shape rather than collapsing to the sign-marginal
    # probability first). Higher (less negative / larger negentropy) =
    # more peaked posterior = more confident.
    entropy_c = np.sum(post * np.log(post + 1e-12), axis=1)   # = -H(post)

    # --- chosen/unchosen pool classification + activity, on the FULL
    # population (this defines the animal's own, full-information
    # PartialPositiveEvidence confidence computation) ---
    rpref_full, lpref_full = _classify_pools(z, choice)
    x_full = np.nanmean(z[:, rpref_full], 1) if rpref_full.sum() else np.full(ntr, np.nan)
    y_full = np.nanmean(z[:, lpref_full], 1) if lpref_full.sum() else np.full(ntr, np.nan)

    # (viii) Partial Positive Evidence: chosen/unchosen pool-weighting model
    # that UNDERWEIGHTS the unchosen pool by a free factor alpha in [0,1],
    # built from the pool means of the FULL simulated population (x = chosen-
    # preferring pool mean activity, y = other pool) -- not a reconstruction
    # from |e|, unlike the behavior-only analog in
    # entropy_behavioral_peb_test/fit_free_param_models.py's
    # PartialPositiveEvidence, which has no access to real per-neuron pools
    # and must reconstruct an analogous total-pool-drive constant from the
    # fitted SDT model instead.
    #   chosen   = x if e>0 else y     (pool preferring the CHOSEN side)
    #   unchosen = y if e>0 else x
    #   partial_pos(alpha) = chosen - alpha*unchosen
    # (SUBTRACTED, not added -- corrected from an earlier draft of this
    # readout, which added alpha*unchosen and consequently had no calibrated
    # limit anywhere in alpha in [0,1]. With subtraction: alpha=1 gives
    # chosen-unchosen, i.e. the same signed contrast the linear decision
    # variable e is built from -- the fully rational, calibrated limit;
    # alpha=0 gives the chosen pool alone ("pure" positive evidence,
    # disconfirming activity fully discounted, magnitude-conflating). alpha
    # in (0,1) is a genuine partial discount of the unchosen pool relative
    # to the fully-rational alpha=1 case, matching the "underweighted, not
    # ignored" framing this readout is meant to test.)
    chosen_pool = np.where(e > 0, x_full, y_full)
    unchosen_pool = np.where(e > 0, y_full, x_full)
    partial_pos_c = chosen_pool - 0.5 * unchosen_pool

    # --- RECORDED subsample: classify chosen/unchosen pools and build d,g
    # from ONLY this subsample's activity, exactly as a real experiment
    # would (module docstring item (2)). This is what the downstream (a,b)
    # pool-weighting regression in sim_peb_vs_me.py actually uses, via the
    # returned counts/rpref/lpref/choice -- the seven targets above stay
    # full-population quantities, so the regression tests whether recording
    # only a small subsample of a much larger decision population can, by
    # itself, produce a spurious chosen-pool-dominated (a,b) pattern for
    # read-outs that are not choice-asymmetric by construction. ---
    counts_rec = counts[:, rec_idx]
    z_rec = zscore_cols(counts_rec)
    rpref, lpref = _classify_pools(z_rec, choice)
    if rpref.sum() < MIN_GRP or lpref.sum() < MIN_GRP:
        return None

    x = np.nanmean(z_rec[:, rpref], 1)
    y = np.nanmean(z_rec[:, lpref], 1)
    d = x - y; g = x + y
    d -= np.nanmean(d); g -= np.nanmean(g)

    return dict(d=d, g=g, choice=choice, acc=acc,
                casandre=casandre_c, bayes=bayes_c, selfcon=selfcon_c,
                fullbayes=fullbayes_c, blind=blind_c, entropy=entropy_c,
                chosen_pool=chosen_pool, unchosen_pool=unchosen_pool,
                partial_pos=partial_pos_c,
                s=s, ci=ci, e=e,
                # exposed for the PEB-vs-ME analysis (per-choice E+/E- regression),
                # which needs the RECORDED subsample's raw counts + neuron-preference
                # masks (re-z-scored within each choice separately downstream) --
                # NOT the full population used to compute the targets above.
                counts=counts_rec, rpref=rpref, lpref=lpref)


# ----------------------------------------------------------------------------
# Map building (faithful to functions/build_pub_maps.m + gauss2d_smooth.m)
# ----------------------------------------------------------------------------
def gauss2d_smooth(M, sigma):
    r = int(np.ceil(3*sigma))
    ax = np.arange(-r, r+1)
    k1 = np.exp(-(ax**2)/(2*sigma**2)); k1 /= k1.sum()
    K = np.outer(k1, k1)
    from scipy.signal import convolve2d
    num = convolve2d(np.nan_to_num(M), K, mode='same')
    return num

def smooth_frac(num, den, sigma):
    from scipy.signal import convolve2d
    r = int(np.ceil(3*sigma)); ax = np.arange(-r, r+1)
    k1 = np.exp(-(ax**2)/(2*sigma**2)); k1 /= k1.sum(); K = np.outer(k1, k1)
    sn = convolve2d(num, K, mode='same')
    sd = convolve2d(den, K, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = sn / sd
    return frac, sd

def build_binned(d, g, values, mask=None, edges=None):
    """Return smoothed P(value|d,g), smoothed trial count. values in {0,1} or NaN.
    Vectorized binning. If edges given, reuse them (for bootstrap consistency)."""
    if edges is None:
        dmin, dmax = d.min()-PAD, d.max()+PAD
        gmin, gmax = g.min()-PAD, g.max()+PAD
        de = np.linspace(dmin, dmax, GRID_BINS+1)
        ge = np.linspace(gmin, gmax, GRID_BINS+1)
    else:
        de, ge = edges
    di = np.clip(np.digitize(d, de)-1, 0, GRID_BINS-1)
    gi = np.clip(np.digitize(g, ge)-1, 0, GRID_BINS-1)
    num = np.zeros((GRID_BINS, GRID_BINS)); den = np.zeros((GRID_BINS, GRID_BINS))
    v = values
    sel = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
    np.add.at(den, (gi[sel], di[sel]), 1.0)
    np.add.at(num, (gi[sel], di[sel]), v[sel])
    frac, sden = smooth_frac(num, den, SMOOTH_SIGMA)
    return frac, sden, (de, ge)


# ----------------------------------------------------------------------------
# Similarity metrics (faithful to functions/similarity_matrix.m + trial_axis_vectors.m)
# ----------------------------------------------------------------------------
def wcorr(a, b, wt):
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(wt) & (wt > 0)
    a, b, wt = a[m], b[m], wt[m]
    ma = np.sum(wt*a)/np.sum(wt); mb = np.sum(wt*b)/np.sum(wt)
    cov = np.sum(wt*(a-ma)*(b-mb))
    va = np.sum(wt*(a-ma)**2); vb = np.sum(wt*(b-mb)**2)
    return cov/np.sqrt(va*vb)

def wr2(y, x, wt):
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(wt) & (wt > 0)
    y, x, wt = y[m], x[m], wt[m]
    W = np.diag(wt); X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(X.T@W@X, X.T@W@y)
    yhat = X@beta
    my = np.sum(wt*y)/np.sum(wt)
    ss_res = np.sum(wt*(y-yhat)**2); ss_tot = np.sum(wt*(y-my)**2)
    return 1 - ss_res/ss_tot

def axis_vec(d, g, val):
    m = np.isfinite(val)
    dz = (d[m]-d[m].mean())/d[m].std(); gz = (g[m]-g[m].mean())/g[m].std()
    X = np.column_stack([np.ones(m.sum()), dz, gz])
    beta = np.linalg.lstsq(X, val[m], rcond=None)[0][1:]
    return beta/np.linalg.norm(beta)

def cos_sim(u, v):
    return float(np.dot(u, v))
