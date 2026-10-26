#!/usr/bin/env python3
"""
Free-parameter behavioral (SDT) confidence models, fit to REAL per-trial
choice/confidence data -- no neural spikes, no population simulation, and
(as of this version) no neural data of ANY kind, not even indirectly: data
now comes from build_real_behavior.py / data/real_behavior_full.csv, a
from-scratch extraction of ONLY the small per-session behavioral
trialMatrix (real, signed orientation in degrees; real, continuous
contrast; real choice/correct/confidence) from
masterNeuralStruct_TEST.mat's lightweight metadata -- never the per-neuron
spike-count arrays, and never the cross-validated neural pool-projection
pipeline that analysis_chatGPT/analysis_pipeline.py's
crossfit_trial_pool_scores.csv (used by earlier versions of this script,
and still used by real_data_test.py / Section 2's `g` proxy) depends on.
See build_real_behavior.py's docstring for the exact column provenance.
This also means `s` below is real |orientation| in degrees, not a
per-session z-score -- so all figure x-axes are now in true degrees, and
s=0 is a genuine, physically meaningful reference point (an objectively
ambiguous stimulus), not an arbitrary z-scored mean.

Model (per animal, fit separately):
  Choice/accuracy (shared across all confidence models):
    e = k*s + sigma_c*eta,   eta ~ N(0,1),   s = |orientation| (degrees)
    choice correct  <=>  e > 0
    sigma_c = sigma_hi (c=high contrast, fixed at 1 as reference) or
              sigma_lo (c=low contrast, free); contrast affects RELIABILITY
              (noise), not mean signal strength -- the standard SDT reading
              of "contrast" as a reliability/precision manipulation.
    Free params: k, sigma_lo ONLY -- no intercept/bias term. P(correct) is
    forced through exactly 0.5 at s=0 for BOTH contrast conditions, which is
    not an assumption but a verified fact about this data: orientation=0
    trials have a correct side assigned by an independent random draw (see
    build_real_behavior.py / the raw trialMatrix column documentation), so
    no amount of real response bias can move expected accuracy away from
    0.5 there -- confirmed directly on raw (unbinned) trials: 0.51/0.49
    (Ziggy hi/lo) and 0.50/0.53 (Friedrich hi/lo), both within noise of
    0.5. An earlier version of this script gave the accuracy model free
    per-condition intercepts (bias_hi, bias_lo) to fix a z-scored-data
    issue (the old z-scoring's zero point wasn't the true chance point);
    once the data source switched to real degrees that justification no
    longer applied, but the free intercepts were left in and quietly
    absorbed ~0.1-0.3 of spurious P(correct) shift at s=0 -- a modeling
    artifact with no basis in the real per-trial data, caught by directly
    checking accuracy on literal orientation=0 trials. Removed.
    Fit by probit MLE on real (s, contrast, correct): P(correct) = Phi(k*s/sigma_c).

  Confidence. Two families:

  (1) "Perceptual" read-outs: Type-2 SDT, i.e. confidence is a soft
      threshold on the SAME trial's realized |e| (so it DOES discount for
      trial-specific errors -- an error with a given |e| gets a different
      predicted confidence than a correct trial with the same |e|, because
      |e|'s distribution is truncated differently by outcome):
        Calibrated   : p_conf = Phi(kappa * |e|/sigma_c)             [TRUE per-trial sigma_c]
        ContrastBlind: p_conf = Phi(kappa * |e|/sigma_pooled)        [a single, condition-blind sigma]
      One free parameter (kappa) each (+bias), fit by MLE using a Monte
      Carlo expectation over the outcome-truncated distribution of |e|
      (2000 draws per unique (s, contrast, outcome) cell, cached and reused
      across optimizer iterations and for the plotted curve).

  (1b) "Entropy(e)": p_conf = Phi(kappa * negH + bias), where negH is the
      NEGATIVE SHANNON ENTROPY OF THE FULL (sign x |orientation|-bin x
      contrast) POSTERIOR that FullBayes(e) below computes internally,
      sum_h post(h)*log(post(h)), WITHOUT first collapsing that posterior
      to a scalar P(chosen side correct). This -- not a binary entropy of
      a single pooled-reliability probability -- is the genuine
      behavior-only analog of Fig. 5's simulated Entropy read-out (Methods,
      sec:simpeb-methods, item (v): "the negative Shannon entropy of that
      same full posterior"), since it is built from literally the same
      full joint posterior FullBayes(e) uses, merely summarized by its
      overall distributional sharpness instead of by sign-specific mass.
      An earlier version of this script instead computed p_conf = Phi(kappa
      * (-H(Phi(|e|/sigma_pooled))) + bias), the negative BINARY entropy of
      a single scalar probability derived from a condition-blind sigma --
      a structurally different, much cheaper quantity that discards
      exactly the multivariate-posterior information the manuscript's own
      explanation for Entropy's simulated behavior (\"a distributional
      summary of the whole population posterior\") depends on. Corrected
      here to reuse FullBayes(e)'s actual grid/posterior
      (fullbayes_e_neg_entropy), so the behavioral fit and the simulated
      Entropy read-out are now the same computation applied to two
      different data sources, as the manuscript text claims. Two free
      parameters (kappa, bias), fit the same way as (1)'s models but
      requiring the fitted FullBayes(e) hypothesis grid (built once per
      animal, see fit_animal) rather than a scalar sigma_ref.

  (2) "PositiveEvidence" [RETIRED]: an earlier version of this module fit
      a model with confidence as a function of the NOMINAL/objective
      stimulus (s, contrast) alone, Phi(kappa_s*s + kappa_c*hi + bias),
      never touching the realized |e| or outcome of that trial. It is
      retired here: it was conceived as a nominal-stimulus stand-in for
      "pure positive evidence" (confidence blind to whether a given
      decision was actually correct), but item (7) below shows that this
      target -- discounting the unchosen pool entirely -- is not a
      separately identifiable claim once chosen/unchosen pool activity is
      reconstructed from a single scalar decision variable plus a
      per-condition constant. See item (7) for the model that replaces it.

  (3) "CASANDRE": confidence = Phi(kappa * |e|/sigma_hat + bias), where
      sigma_hat = sigma_c * exp(tau * z), z ~ N(0,1) drawn fresh per trial,
      independent of e -- i.e. confidence uses the TRUE per-condition
      sigma_c (unlike ContrastBlind/Entropy(e)) but through a noisy, unbiased
      (in log-space) per-trial ESTIMATE of it, with the noise scale tau a
      free parameter. This is the one model here that genuinely needs a
      second Monte Carlo dimension (independent of the |e| MC cache): for
      each cell, e_abs (n_mc draws, outcome-truncated, already cached) is
      paired with an independent draw of z from a fixed, shared standard
      normal vector (META_Z), so E[Phi(kappa*|e|/sigma_hat+bias)] is a valid
      Monte Carlo estimate over the (e, sigma_hat) product distribution.
      Fit by MLE over (kappa, bias, tau). This is the free-parameter
      behavioral analog of Fig. 5's CASANDRE read-out -- included for
      completeness alongside Calibrated/ContrastBlind/Entropy(e). Note
      CASANDRE remains built on the TRUE sigma_c, so
      unlike ContrastBlind/Entropy(e) it is not expected to show the
      paradoxical PEB sign flip (see Fig. 5 / Section 1: CASANDRE reproduces
      the calibrated, not PEB-like, sign in the simulation); it is a "noisy
      but unbiased" control, not a magnitude-conflating mechanism.

  (4) "Bayesian(|e|)": the ZERO-free-parameter special case of Calibrated
      (kappa=1, bias=0 fixed, not fit) -- the literal normative posterior
      P(correct)=Phi(|e|/sigma_c), with no report-mapping allowed at all.
      Reported alongside Calibrated (2 free params) so the gap between them
      shows how much of Calibrated's fit improvement comes from genuinely
      better nll vs. just having more free parameters to move around.

  (5) "FullBayes(e)": a behavior-only analog of Fig. 5's sixth read-out,
      FullBayes(r). The manuscript's FullBayes(r) infers BOTH the stimulus
      magnitude AND the contrast condition jointly from the full per-neuron
      population response r (it does not assume the trial's contrast is
      known, unlike Bayesian(|e|)/CASANDRE/Calibrated, which all condition
      on the true, known sigma_c). That is genuinely a different assumption,
      not just a different data source, and it can be reproduced WITHOUT any
      neural data at all by doing the same joint inference on the scalar e
      instead of the raw population vector r: given a trial's realized e
      (reconstructed sign-canonically from the outcome-truncated |e| MC
      cache: e = +|e| if correct, -|e| if error, since s is defined
      canonically positive), the model evaluates the Gaussian likelihood of
      e under EVERY hypothesis in a grid of (stimulus-sign x |orientation|
      magnitude-bin x contrast condition), using the SAME fitted (k,
      sigma_hi, sigma_lo, bias_hi, bias_lo) as every other model here (i.e.
      "reasonable model-based assumptions" = the same SDT generative model
      already estimated from real accuracy, not a new fabricated neuron
      population), with a flat prior over sign and over the |orientation|
      magnitude grid (the same 40 quantile-bin centers used elsewhere) and
      the empirical hi/lo contrast frequency as the prior over contrast.
      Confidence = the posterior mass on whichever sign was actually chosen,
      marginalizing out (summing over) the unknown contrast, exactly
      mirroring FullBayes(r)'s p_pos/p_neg construction. Zero free
      parameters, like Bayesian(|e|) -- it isn't fit to confidence data at
      all, only evaluated, so it's reported the same way (nll only, no
      kappa/bias). This is the FullBayes(r)-vs-Bayesian comparison from
      Fig. 5 (does inferring contrast jointly rather than assuming it known
      matter?) reproduced entirely from real behavior.

  (6) "Bayesian(crit)" / "FullBayes(crit)": Bayesian(|e|) and FullBayes(e)
      are both hard-wired to assume the animal's binary high/low-confidence
      REPORT literally equals the ideal-observer POSTERIOR PROBABILITY of
      being correct -- there is no free criterion at all, so if real
      report rates differ systematically from that posterior (they do:
      both models are far too liberal, over-predicting "high confidence"
      -- see Bottom line / results_summary.md), the zero-parameter model
      has no way to correct for it. These two variants add exactly one
      free parameter -- a criterion/bias, SHARED across both contrast
      conditions (not fit per-condition) -- on top of the SAME normative
      computation, with the sensitivity kappa still fixed at 1 (so it
      remains "the same normative sensitivity as the ideal observer, just
      with the animal's own report threshold"):
        Bayesian(crit) : p_conf = Phi(|e|/sigma_c + bias)         [same feature as Bayesian(|e|), kappa fixed]
        FullBayes(crit): p_conf = Phi(Phi^-1(p_hyp) + bias)       [p_hyp = FullBayes(e)'s raw posterior,
                         probit-recalibrated by a single additive shift in z-space]
      One free parameter (bias only) each. Comparing these to Bayesian(|e|)/
      FullBayes(e) (0 params) and to Calibrated (2 params: kappa AND bias)
      isolates how much of the ideal-observer's overconfidence is fixed by
      a criterion shift ALONE, vs. how much also needs kappa (sensitivity)
      to move.

  (7) "PositiveEvidence/Gain": reconstructs chosen/unchosen pool activity
      from the outcome-truncated |e| of a trial and a per-condition
      total-pool-drive constant g_c via the standard 2-pool identity
      e = chosen - unchosen, g_c = chosen + unchosen (chosen=(g_c+|e|)/2,
      unchosen=(g_c-|e|)/2 -- the same E+/E- decomposition used elsewhere
      in this repo, e.g. the g=E++E- construction used in
      real_data_test.py, reconstructed here from the scalar SDT model
      since no real per-neuron pools exist in this behavior-only
      framework), and asks whether confidence underweights the unchosen
      pool relative to the chosen one:
        p_conf = Phi(kappa * [chosen - alpha*unchosen] + bias)
               = Phi(kappa*(1+alpha)/2 * |e| + [kappa*g_c*(1-alpha)/2 + bias])
      with g_hi=1 fixed as reference and g_lo free (one value per contrast
      condition).

      THIS PARAMETRIZATION IS NOT IDENTIFIABLE FROM CONFIDENCE DATA ALONE,
      and that is the interesting finding here, not a nuisance to engineer
      around. Write Q = kappa*(1+alpha)/2 (the coefficient on |e|, shared
      across conditions since kappa and alpha are both global) and
      A_c = kappa*g_c*(1-alpha)/2 + bias (the per-condition intercept, c
      in {hi, lo}). The confidence likelihood depends on (kappa, bias,
      alpha, g_lo) ONLY through (Q, A_hi, A_lo) -- three numbers -- because
      g_c and alpha never appear separately anywhere in the model, only as
      the single product kappa*g_c*(1-alpha)/2 inside A_c. Any rescaling of
      alpha with a compensating change in g_lo that leaves that product
      (and Q) unchanged leaves every prediction, and hence the likelihood,
      EXACTLY unchanged -- confirmed numerically to 4 decimal places in
      both animals (see diagnose_alpha_unconstrained.py). The two
      interpretations this ambiguity spans are exactly the two halves of
      this model name: at alpha=1, the (1-alpha) factor vanishes and g_c
      drops out of A_c entirely, so the whole between-condition intercept
      difference becomes a pure per-condition criterion/GAIN shift, with
      no claim that the unchosen pool is discounted at all; at any
      alpha < 1 (down toward -1), part of that same intercept difference
      is instead attributed to a genuine POSITIVE-EVIDENCE discount of the
      unchosen pool, with g_lo adjusted to compensate. Every point along
      that trade-off reproduces the identical fit, so nothing in the
      confidence data can adjudicate between "the low-contrast confidence
      criterion is simply different" and "the unchosen pool is partially
      discounted, and separately its drive is reduced at low contrast" --
      hence the compound name, and hence why alpha and g_lo are NOT
      reported as separately fitted quantities below.

      What is fit and reported is only the three quantities the data
      actually identify, directly: Q (shared |e|-slope) and two free,
      independent per-condition intercepts bias_hi, bias_lo --
        p_conf = Phi(Q*|e| + bias_hi)   [high contrast]
        p_conf = Phi(Q*|e| + bias_lo)   [low contrast]
      fit by the same MLE/ridge procedure as Calibrated/ContrastBlind/
      Entropy(e) (a weak L2 ridge on (Q, bias_hi, bias_lo), same strength
      as L2_PENALTY there). This supersedes an earlier 4-parameter
      (kappa, bias, g_lo, alpha) version of this fit -- still reproducible
      via diagnose_alpha_unconstrained.py, where the identifiability check
      above was first established -- and also retires the separate
      "PositiveEvidence" model above, whose own concept (a pure, total
      discount of the unchosen pool) is just the alpha=0 point of the very
      trade-off this section shows the data cannot distinguish from every
      other point on it, including the pure-gain endpoint alpha=1.

  (8) MODEL COMPARISON BY AIC/BIC, not raw nll: the nine models above have
      between 0 and 3 free parameters, so ranking them by nll alone favors
      the more flexible ones for free. AIC = 2*nparams + 2*nll and
      BIC = nparams*log(n_trials) + 2*nll are reported alongside nll for
      every model (see MODEL_NPARAMS below for the exact per-model count
      used, and __main__ for the printed comparison table); lower is
      better for both. BIC's per-parameter penalty (log(n_trials), n_trials
      in the tens of thousands here, so log(n) ~ 10.5) is much harsher than
      AIC's flat 2-per-parameter penalty, so the two criteria can disagree
      about whether a 1-2 parameter difference is "worth it" even when nll
      alone would call it decisively.
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import norm
from scipy.optimize import minimize

RNG = default_rng(0)
META_RNG = default_rng(1)   # separate, independent stream for CASANDRE's meta-noise
N_MC = 4000
META_Z = META_RNG.standard_normal(N_MC)   # fixed across cells/optimizer iters (common random numbers)
EPS = 1e-9
L2_PENALTY = 2e-3  # weakly-informative ridge on (kappa, bias) for the |e|-based
                   # models; prevents divergence-to-a-step-function that
                   # unpenalized probit MLE shows when a model fits the data
                   # poorly (kappa -> inf, bias -> -inf at a fixed ratio).


def load(csv_path="data/real_behavior_full.csv"):
    """Real per-trial behavior only -- see build_real_behavior.py. No neural
    data, no z-scoring: "absori" here is TRUE |orientation| in degrees, and
    "hi"/"lo" (high/low contrast) is the trial matrix's OWN categorical
    contrast-level index ("contrast_id", column 12 of the 31-column trial
    matrix, the same field fig3_cross_decoder/fig4_peb_pools/
    si_stimulus_difficulty_and_cv_controls use directly as "contrast_id"),
    not a derived split of the raw, near-continuous "contrast" column.
    contrast_id==1 is the high-contrast level (raw contrast ~1.13-1.27,
    65.3% of trials) and contrast_id==2 is low (~0.44-0.78).

    An earlier version of this function instead used a per-animal MEDIAN
    split of the raw "contrast" column. That is NOT equivalent to this
    categorical split: the median (1.20, both animals) falls just inside
    the high-contrast band rather than in the gap between the two bands,
    so ~15.4% of trials (3,203 Ziggy, 2,525 Friedrich) were previously
    assigned to the wrong contrast condition. Fixed after an external
    reproducibility check caught the mismatch by cross-referencing this
    file's continuous "contrast" column against the categorical field
    used elsewhere in the codebase."""
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["animal"] = df["animal"].str.capitalize()
    df["absori"] = np.abs(df["orientation"])
    df["hi"] = (df["contrast_id"] == 1).astype(int)
    return df


# ---------------------------------------------------------------- accuracy --
def neg_ll_accuracy(theta, s, hi, correct):
    k, log_sigma_lo = theta
    sigma_lo = np.exp(log_sigma_lo)
    sigma_c = np.where(hi == 1, 1.0, sigma_lo)
    p = norm.cdf(k * s / sigma_c)
    p = np.clip(p, EPS, 1 - EPS)
    return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).sum()


def fit_accuracy(s, hi, correct):
    # NO free intercept, for either condition: P(correct)=Phi(k*s/sigma_c)
    # is forced through exactly 0.5 at s=0 for both. This is a verified
    # fact about the real data, not a modeling convenience -- see module
    # docstring: real accuracy on literal orientation=0 trials is ~0.5 in
    # both contrast conditions in both animals, because those trials have
    # an independently, randomly assigned correct side. (An earlier
    # version of this function had free bias_hi/bias_lo, left over from
    # when "s" was a z-scored quantity whose zero point was NOT the true
    # chance point -- that justification no longer applies now that s is
    # real degrees, and the free intercepts were absorbing a spurious,
    # data-unsupported shift. Removed.)
    res = minimize(neg_ll_accuracy, x0=[1.0, 0.0], args=(s, hi, correct),
                    method="Nelder-Mead",
                    options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
    k, log_sigma_lo = res.x
    sigma_lo = np.exp(log_sigma_lo)
    return dict(k=k, sigma_hi=1.0, sigma_lo=sigma_lo, bias_hi=0.0, bias_lo=0.0,
                nll=res.fun, ok=res.success)


# --------------------------------------------------- |e| Monte Carlo cache --
def mc_abs_e_cache(unique_cells, k, sigma_hi, sigma_lo, bias_hi=0.0, bias_lo=0.0,
                    n_mc=N_MC, rng=RNG):
    """unique_cells: iterable of (s, hi, outcome) -> dict[(s,hi,outcome)] = |e| samples (n_mc,)
    mean is k*s + bias_c*sigma_c so that mean/sigma_c = k*s/sigma_c + bias_c,
    matching the accuracy model's P(correct) = Phi(k*s/sigma_c + bias_c)."""
    cache = {}
    for (s, hi, outcome) in unique_cells:
        sigma_c = sigma_hi if hi == 1 else sigma_lo
        bias_c = bias_hi if hi == 1 else bias_lo
        mean = k * s + bias_c * sigma_c
        a, b = (-mean / sigma_c, np.inf) if outcome == 1 else (-np.inf, -mean / sigma_c)
        lo_cdf, hi_cdf = norm.cdf(a), norm.cdf(b)
        u = rng.uniform(lo_cdf, hi_cdf, size=n_mc)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        z = norm.ppf(u)
        e = mean + sigma_c * z
        cache[(s, hi, outcome)] = np.abs(e)
    return cache


# --------------------------------------------------------------- confidence -
# "Bayesian(|e|)" is the zero-free-parameter special case of Calibrated
# (kappa=1, bias=0: the literal calibrated posterior P(correct)=Phi(|e|/sigma_c),
# with no additional report-mapping/miscalibration allowed) -- same sigma_c
# reference and same feature (|e|/sigma_c) as Calibrated, so it reuses both
# sigma_ref_for and confidence_feature under the "Calibrated" branch.
E_MODELS = ("Calibrated", "ContrastBlind")   # |e|-based, outcome-sensitive, fixed sref
# "Entropy(e)" is fit separately (fit_entropy_e, below) since it needs
# FullBayes(e)'s hypothesis grid rather than a scalar sigma_ref -- see
# module docstring (1b) and fit_animal.
ALL_MODELS = ("Calibrated", "Bayesian(|e|)", "Bayesian(crit)", "ContrastBlind", "Entropy(e)",
              "CASANDRE", "FullBayes(e)", "FullBayes(crit)",
              "PositiveEvidence/Gain")

# Free-parameter count per model, for AIC/BIC (see module docstring section 8).
# Counts confidence-model parameters only (the shared accuracy fit -- k,
# sigma_lo -- is common to every model and so does not change their relative
# ranking; it is left out here, matching standard practice when comparing
# nested/non-nested models that all share the same upstream stage).
MODEL_NPARAMS = {
    "Calibrated": 2,               # kappa, bias
    "Bayesian(|e|)": 0,            # kappa=1, bias=0 fixed
    "Bayesian(crit)": 1,           # bias (kappa fixed at 1)
    "ContrastBlind": 2,            # kappa, bias
    "Entropy(e)": 2,                # kappa, bias
    "CASANDRE": 3,                 # kappa, bias, tau
    "FullBayes(e)": 0,             # evaluated only, not fit
    "FullBayes(crit)": 1,          # bias
    "PositiveEvidence/Gain": 3,    # Q, bias_hi, bias_lo (see module docstring item (7))
}


def aic(nll, nparams):
    """AIC = 2*nparams + 2*nll (nll is the NEGATIVE log-likelihood already, so
    this is the usual -2*logL + 2*k written in terms of nll directly). Lower
    is better."""
    return 2.0 * nparams + 2.0 * nll


def bic(nll, nparams, n_trials):
    """BIC = nparams*log(n_trials) + 2*nll. Lower is better; the per-parameter
    penalty (log(n_trials)) is much harsher than AIC's flat 2 when n_trials is
    in the tens of thousands, as it is here (log(n) ~ 10.5)."""
    return nparams * np.log(n_trials) + 2.0 * nll


def sigma_ref_for(model, hi, sigma_hi, sigma_lo, sigma_pooled):
    if model in ("Calibrated", "Bayesian(|e|)", "Bayesian(crit)"):
        return sigma_hi if hi == 1 else sigma_lo
    if model == "ContrastBlind":
        return sigma_pooled
    raise ValueError(model)


def confidence_feature(model, e_abs, sref):
    """The quantity kappa multiplies (plus bias, through Phi) for each scalar-
    sigma_ref |e|-based model (Calibrated/Bayesian(|e|)/Bayesian(crit)/
    ContrastBlind use the raw evidence ratio). Entropy(e) is NOT included here
    -- it needs FullBayes(e)'s full hypothesis grid, not a scalar sref, so it
    is fit and evaluated separately (fullbayes_e_neg_entropy / fit_entropy_e,
    below)."""
    if model in ("Calibrated", "Bayesian(|e|)", "Bayesian(crit)", "ContrastBlind"):
        return e_abs / sref
    raise ValueError(model)


def fit_bayesian_abs_e(cache, sigma_hi, sigma_lo, sigma_pooled, n_cell_trials, k_conf_sum):
    """Zero free parameters: kappa=1, bias=0 fixed -- the literal normative
    calibrated posterior, not fit to the data at all. Its nll is still worth
    reporting alongside Calibrated's (2 free params) as a check on how much
    real confidence reports deviate from the ideal-observer report."""
    nll = neg_ll_e_confidence([1.0, 0.0], cache, "Calibrated", sigma_hi, sigma_lo,
                              sigma_pooled, k_conf_sum, n_cell_trials)
    return dict(kappa=1.0, bias=0.0, nll=nll, ok=True)


def fit_bayesian_crit(cache, sigma_hi, sigma_lo, sigma_pooled, n_cell_trials, k_conf_sum):
    """Bayesian(crit): same feature/reference as Bayesian(|e|) -- kappa fixed
    at 1, true per-condition sigma_c -- but with ONE free, condition-
    independent criterion/bias parameter added. Isolates whether a pure
    criterion shift (not a sensitivity rescaling) accounts for the
    zero-parameter ideal observer's overconfidence."""
    def negll(theta):
        return neg_ll_e_confidence([1.0, theta[0]], cache, "Calibrated", sigma_hi, sigma_lo,
                                    sigma_pooled, k_conf_sum, n_cell_trials)
    res = minimize(negll, x0=[0.0], method="Nelder-Mead",
                    options=dict(xatol=1e-8, fatol=1e-8, maxiter=2000))
    return dict(kappa=1.0, bias=res.x[0], nll=res.fun, ok=res.success)


def neg_ll_e_confidence(theta, cache, model, sigma_hi, sigma_lo, sigma_pooled,
                         k_conf_sum, n_cell_trials):
    kappa, bias = theta
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        sref = sigma_ref_for(model, hi, sigma_hi, sigma_lo, sigma_pooled)
        feat = confidence_feature(model, cache[cell], sref)
        p_bar = np.clip(norm.cdf(kappa * feat + bias).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    total += L2_PENALTY * (kappa ** 2 + bias ** 2)
    return total


def fit_e_confidence_model(model, cache, sigma_hi, sigma_lo, sigma_pooled,
                            n_cell_trials, k_conf_sum):
    res = minimize(neg_ll_e_confidence, x0=[1.0, 0.0],
                    args=(cache, model, sigma_hi, sigma_lo, sigma_pooled,
                          k_conf_sum, n_cell_trials),
                    method="L-BFGS-B", bounds=[(-15, 15), (-15, 15)])
    kappa, bias = res.x
    return dict(kappa=kappa, bias=bias, nll=res.fun, ok=res.success)


def neg_ll_casandre(theta, cache, sigma_hi, sigma_lo, n_cell_trials, k_conf_sum):
    kappa, bias, tau = theta
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        sigma_c = sigma_hi if hi == 1 else sigma_lo
        sigma_hat = sigma_c * np.exp(tau * META_Z)
        feat = cache[cell] / sigma_hat
        p_bar = np.clip(norm.cdf(kappa * feat + bias).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    total += L2_PENALTY * (kappa ** 2 + bias ** 2) + 1e-3 * tau ** 2
    return total


def fit_casandre(cache, sigma_hi, sigma_lo, n_cell_trials, k_conf_sum):
    res = minimize(neg_ll_casandre, x0=[1.0, 0.0, 0.3],
                    args=(cache, sigma_hi, sigma_lo, n_cell_trials, k_conf_sum),
                    method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
    kappa, bias, tau = res.x
    return dict(kappa=kappa, bias=bias, tau=abs(tau), nll=res.fun, ok=res.success)


def fullbayes_e_grid(k, sigma_hi, sigma_lo, bias_hi, bias_lo, Sk, p_hi_prior):
    """Flat hypothesis grid for FullBayes(e): sign in {+1,-1} x |orientation|
    magnitude bin Sk x contrast in {hi, lo}. Returns (means, sds, priors,
    hyp_signs), each shape (2*len(Sk)*2,). Prior is flat over sign and over
    the (already roughly-equal-count, quantile-binned) Sk grid, and equal to
    the empirical hi/lo contrast frequency over contrast -- "reasonable
    model-based assumptions", not fitted, not from neural data."""
    Sk = np.asarray(Sk, dtype=float)
    Sk = Sk[np.isfinite(Sk)]
    K = len(Sk)
    conds = ((sigma_hi, bias_hi, p_hi_prior), (sigma_lo, bias_lo, 1.0 - p_hi_prior))
    means, sds, priors, hyp_signs = [], [], [], []
    for sign in (1.0, -1.0):
        for sigma_c, bias_c, pc in conds:
            means.append(sign * (k * Sk + bias_c * sigma_c))
            sds.append(np.full(K, sigma_c))
            priors.append(np.full(K, 0.5 * pc / K))
            hyp_signs.append(np.full(K, sign))
    means = np.concatenate(means); sds = np.concatenate(sds)
    priors = np.concatenate(priors); priors = priors / priors.sum()
    hyp_signs = np.concatenate(hyp_signs)
    return means, sds, priors, hyp_signs


def fullbayes_e_readout(e_signed, grid):
    """e_signed: (n,) samples (canonical sign convention: +|e| if correct,
    -|e| if error). Returns (n,) predicted P(chosen side correct), i.e. the
    posterior mass on whichever sign matches sign(e_signed), marginalizing
    over the unknown contrast condition -- FullBayes(r)'s p_pos/p_neg
    construction, reproduced from the scalar e alone."""
    means, sds, priors, hyp_signs = grid
    e = np.asarray(e_signed)[:, None]
    logpost = norm.logpdf(e, loc=means[None, :], scale=sds[None, :]) + np.log(priors)[None, :]
    logpost -= logpost.max(1, keepdims=True)
    post = np.exp(logpost)
    post /= post.sum(1, keepdims=True)
    p_pos = (post * (hyp_signs[None, :] > 0)).sum(1)
    return np.where(e_signed >= 0, p_pos, 1.0 - p_pos)


def fullbayes_e_neg_entropy(e_signed, grid):
    """Entropy(e)'s feature: the negative Shannon entropy of the SAME full
    posterior fullbayes_e_readout computes, sum_h post(h)*log(post(h)),
    WITHOUT collapsing it to P(chosen side correct) first. This is the
    scalar-e analog of Fig. 5's simulated Entropy read-out (Methods,
    sec:simpeb-methods, item (v): "the negative Shannon entropy of that same
    full posterior" over the discrete stimulus x contrast grid) -- built from
    literally the same grid/posterior as fullbayes_e_readout/FullBayes(e),
    merely summarized differently. Bounded above by 0 (uniform posterior) and
    growing more negative as the posterior sharpens on a single hypothesis."""
    means, sds, priors, hyp_signs = grid
    e = np.asarray(e_signed)[:, None]
    logpost = norm.logpdf(e, loc=means[None, :], scale=sds[None, :]) + np.log(priors)[None, :]
    logpost -= logpost.max(1, keepdims=True)
    post = np.exp(logpost)
    post /= post.sum(1, keepdims=True)
    post_c = np.clip(post, 1e-300, 1.0)
    return (post_c * np.log(post_c)).sum(1)


def entropy_e_feature_cache(cache, grid, n_cell_trials):
    """Precompute fullbayes_e_neg_entropy once per (s, contrast, outcome) cell,
    since it depends only on the fixed accuracy fit/grid, not on the
    (kappa, bias) being optimized -- reused across every neg_ll_entropy_e call
    during the L-BFGS-B search, matching how the other |e|-based models reuse
    the cached MC |e| samples directly (recomputing the full posterior from
    scratch on every optimizer iteration is the expensive part, not the
    Phi(kappa*feat+bias) step)."""
    feat_cache = {}
    for cell in n_cell_trials:
        s_val, hi, outcome = cell
        e_abs = cache[cell]
        e_signed = e_abs if outcome == 1 else -e_abs
        feat_cache[cell] = fullbayes_e_neg_entropy(e_signed, grid)
    return feat_cache


def neg_ll_entropy_e(theta, feat_cache, n_cell_trials, k_conf_sum):
    """Entropy(e): p_conf = Phi(kappa*negH + bias), negH from the precomputed
    fullbayes_e_neg_entropy feature cache. Two free parameters, fit the same
    way as Calibrated/ContrastBlind (Monte Carlo expectation over the
    outcome-truncated |e| cache), but the feature itself comes from
    FullBayes(e)'s full hypothesis grid rather than a scalar sigma_ref."""
    kappa, bias = theta
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        feat = feat_cache[cell]
        p_bar = np.clip(norm.cdf(kappa * feat + bias).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    total += L2_PENALTY * (kappa ** 2 + bias ** 2)
    return total


def fit_entropy_e(cache, grid, n_cell_trials, k_conf_sum):
    feat_cache = entropy_e_feature_cache(cache, grid, n_cell_trials)
    res = minimize(neg_ll_entropy_e, x0=[1.0, 0.0],
                    args=(feat_cache, n_cell_trials, k_conf_sum),
                    method="L-BFGS-B", bounds=[(-15, 15), (-15, 15)])
    kappa, bias = res.x
    return dict(kappa=kappa, bias=bias, nll=res.fun, ok=res.success)


def neg_ll_fullbayes_e(cache, grid, n_cell_trials, k_conf_sum):
    """Zero free parameters (like Bayesian(|e|)): evaluated, not fit."""
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        e_abs = cache[cell]
        e_signed = e_abs if outcome == 1 else -e_abs
        p_bar = np.clip(fullbayes_e_readout(e_signed, grid).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    return total


def neg_ll_fullbayes_e_crit(theta, cache, grid, n_cell_trials, k_conf_sum):
    """FullBayes(crit): probit-recalibration of FullBayes(e)'s raw posterior
    p_hyp by a single additive shift in z-space (Phi(Phi^-1(p_hyp)+bias)) --
    the FullBayes analog of Bayesian(crit): one free, condition-independent
    criterion parameter on top of the zero-parameter joint-inference posterior."""
    bias = theta[0]
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        e_abs = cache[cell]
        e_signed = e_abs if outcome == 1 else -e_abs
        p_hyp = np.clip(fullbayes_e_readout(e_signed, grid), EPS, 1 - EPS)
        z = norm.ppf(p_hyp) + bias
        p_bar = np.clip(norm.cdf(z).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    return total


def fit_fullbayes_e_crit(cache, grid, n_cell_trials, k_conf_sum):
    res = minimize(neg_ll_fullbayes_e_crit, x0=[0.0],
                    args=(cache, grid, n_cell_trials, k_conf_sum),
                    method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=2000))
    return dict(bias=res.x[0], nll=res.fun, ok=res.success)


def neg_ll_positive_evidence_gain(theta, cache, n_cell_trials, k_conf_sum):
    """Phi(Q*|e| + bias_hi_or_lo): shared slope Q on |e|, free per-condition
    intercept -- the identified content of the chosen/unchosen pool-
    weighting family (module docstring item (7)); not a diagnostic
    reduction of a richer model, but the model itself."""
    Q, bias_hi, bias_lo = theta
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        bias_c = bias_hi if hi == 1 else bias_lo
        e_abs = cache[cell]
        p_bar = np.clip(norm.cdf(Q * e_abs + bias_c).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    total += L2_PENALTY * (Q ** 2 + bias_hi ** 2 + bias_lo ** 2)
    return total


def fit_positive_evidence_gain(cache, n_cell_trials, k_conf_sum):
    """PositiveEvidence/Gain: see module docstring item (7) for why only
    these 3 parameters (Q, bias_hi, bias_lo) are fit and reported, rather
    than the nominal (kappa, bias, g_lo, alpha) of the underlying
    chosen/unchosen pool-weighting family. Multiple restarts (varying
    sign/scale of the initial intercepts) are tried; the lowest-nll result
    across restarts is kept."""
    best = None
    for x0 in ([1.0, 0.0, 0.0], [0.5, -0.5, -0.5], [2.0, 0.5, 0.5],
               [1.0, 1.0, -1.0], [1.0, -1.0, 1.0]):
        res = minimize(neg_ll_positive_evidence_gain, x0=x0,
                        args=(cache, n_cell_trials, k_conf_sum),
                        method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=8000))
        if best is None or res.fun < best.fun:
            best = res
    Q, bias_hi, bias_lo = best.x
    return dict(Q=Q, bias_hi=bias_hi, bias_lo=bias_lo, nll=best.fun, ok=best.success)


def prepare_cells(d, n_sbins=40):
    """Discretize s onto a manageable grid (so the MC cache stays small) and
    aggregate real trial counts per (s_bin, hi, outcome) cell."""
    s = d["absori"].to_numpy()
    edges = np.quantile(s, np.linspace(0, 1, n_sbins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    s_bin_idx = np.digitize(s, edges) - 1
    s_bin_idx = np.clip(s_bin_idx, 0, n_sbins - 1)
    centers = np.array([s[(s_bin_idx == i)].mean() if (s_bin_idx == i).any() else np.nan
                         for i in range(n_sbins)])
    out = d.copy()
    out["s_bin"] = s_bin_idx
    out["s_center"] = centers[s_bin_idx]
    return out


def fit_animal(d_animal, n_sbins=40):
    d = prepare_cells(d_animal, n_sbins=n_sbins)
    s = d["s_center"].to_numpy()
    hi = d["hi"].to_numpy()
    correct = d["correct"].to_numpy()
    conf = d["confidence"].to_numpy()

    acc_fit = fit_accuracy(s, hi, correct)
    k, sigma_hi, sigma_lo, bias_hi, bias_lo = (acc_fit["k"], acc_fit["sigma_hi"],
                                                acc_fit["sigma_lo"], acc_fit["bias_hi"],
                                                acc_fit["bias_lo"])
    sigma_pooled = np.sqrt((sigma_hi ** 2 + sigma_lo ** 2) / 2.0)

    grid_s = np.linspace(np.nanmin(s), np.nanmax(s), 60)
    unique_present = set(zip(np.round(s, 6), hi.astype(int), correct.astype(int)))
    grid_cells = set((float(np.round(gs, 6)), h, o) for gs in grid_s for h in (0, 1) for o in (0, 1))
    all_cells = unique_present | grid_cells
    cache = mc_abs_e_cache(all_cells, k, sigma_hi, sigma_lo, bias_hi=bias_hi, bias_lo=bias_lo)

    cell_key = list(zip(np.round(s, 6), hi.astype(int), correct.astype(int)))
    df_cells = pd.DataFrame({"cell": cell_key, "conf": conf})
    grp = df_cells.groupby("cell")["conf"].agg(["sum", "count"])
    n_cell_trials = grp["count"].to_dict()
    k_conf_sum = grp["sum"].to_dict()

    Sk = np.unique(d["s_center"].dropna().to_numpy())
    p_hi_prior = float(np.mean(hi == 1))
    fb_grid = fullbayes_e_grid(k, sigma_hi, sigma_lo, bias_hi, bias_lo, Sk, p_hi_prior)

    conf_fits = {}
    for model in E_MODELS:
        conf_fits[model] = fit_e_confidence_model(model, cache, sigma_hi, sigma_lo,
                                                   sigma_pooled, n_cell_trials, k_conf_sum)
    conf_fits["Entropy(e)"] = fit_entropy_e(cache, fb_grid, n_cell_trials, k_conf_sum)
    conf_fits["CASANDRE"] = fit_casandre(cache, sigma_hi, sigma_lo, n_cell_trials, k_conf_sum)
    conf_fits["Bayesian(|e|)"] = fit_bayesian_abs_e(cache, sigma_hi, sigma_lo, sigma_pooled,
                                                     n_cell_trials, k_conf_sum)
    conf_fits["Bayesian(crit)"] = fit_bayesian_crit(cache, sigma_hi, sigma_lo, sigma_pooled,
                                                     n_cell_trials, k_conf_sum)
    conf_fits["PositiveEvidence/Gain"] = fit_positive_evidence_gain(cache, n_cell_trials, k_conf_sum)

    fb_nll = neg_ll_fullbayes_e(cache, fb_grid, n_cell_trials, k_conf_sum)
    conf_fits["FullBayes(e)"] = dict(nll=fb_nll, ok=True)
    conf_fits["FullBayes(crit)"] = fit_fullbayes_e_crit(cache, fb_grid, n_cell_trials, k_conf_sum)

    return dict(acc_fit=acc_fit, sigma_pooled=sigma_pooled, conf_fits=conf_fits,
                cache=cache, grid_s=grid_s, k=k, sigma_hi=sigma_hi, sigma_lo=sigma_lo,
                bias_hi=bias_hi, bias_lo=bias_lo, fullbayes_grid=fb_grid)


def predict_confidence_curve(fit, model, hi, outcome):
    """Return (grid_s, predicted P(high conf)) for the fitted model/contrast/outcome."""
    if model == "FullBayes(e)":
        ys = []
        for gs in fit["grid_s"]:
            cell = (float(np.round(gs, 6)), hi, outcome)
            e_abs = fit["cache"][cell]
            e_signed = e_abs if outcome == 1 else -e_abs
            ys.append(fullbayes_e_readout(e_signed, fit["fullbayes_grid"]).mean())
        return fit["grid_s"], np.array(ys)
    if model == "FullBayes(crit)":
        bias = fit["conf_fits"][model]["bias"]
        ys = []
        for gs in fit["grid_s"]:
            cell = (float(np.round(gs, 6)), hi, outcome)
            e_abs = fit["cache"][cell]
            e_signed = e_abs if outcome == 1 else -e_abs
            p_hyp = np.clip(fullbayes_e_readout(e_signed, fit["fullbayes_grid"]), EPS, 1 - EPS)
            z = norm.ppf(p_hyp) + bias
            ys.append(norm.cdf(z).mean())
        return fit["grid_s"], np.array(ys)
    if model == "Entropy(e)":
        cf = fit["conf_fits"][model]
        kappa, bias = cf["kappa"], cf["bias"]
        ys = []
        for gs in fit["grid_s"]:
            cell = (float(np.round(gs, 6)), hi, outcome)
            e_abs = fit["cache"][cell]
            e_signed = e_abs if outcome == 1 else -e_abs
            feat = fullbayes_e_neg_entropy(e_signed, fit["fullbayes_grid"])
            ys.append(norm.cdf(kappa * feat + bias).mean())
        return fit["grid_s"], np.array(ys)
    if model == "PositiveEvidence/Gain":
        cf = fit["conf_fits"][model]
        Q, bias_hi, bias_lo = cf["Q"], cf["bias_hi"], cf["bias_lo"]
        bias_c = bias_hi if hi == 1 else bias_lo
        ys = []
        for gs in fit["grid_s"]:
            cell = (float(np.round(gs, 6)), hi, outcome)
            e_abs = fit["cache"][cell]
            ys.append(norm.cdf(Q * e_abs + bias_c).mean())
        return fit["grid_s"], np.array(ys)
    if model == "CASANDRE":
        cf = fit["conf_fits"][model]
        kappa, bias, tau = cf["kappa"], cf["bias"], cf["tau"]
        sigma_c = fit["sigma_hi"] if hi == 1 else fit["sigma_lo"]
        sigma_hat = sigma_c * np.exp(tau * META_Z)
        ys = []
        for gs in fit["grid_s"]:
            cell = (float(np.round(gs, 6)), hi, outcome)
            feat = fit["cache"][cell] / sigma_hat
            ys.append(norm.cdf(kappa * feat + bias).mean())
        return fit["grid_s"], np.array(ys)
    kappa, bias = fit["conf_fits"][model]["kappa"], fit["conf_fits"][model]["bias"]
    sref = sigma_ref_for(model, hi, fit["sigma_hi"], fit["sigma_lo"], fit["sigma_pooled"])
    ys = []
    for gs in fit["grid_s"]:
        cell = (float(np.round(gs, 6)), hi, outcome)
        feat = confidence_feature(model, fit["cache"][cell], sref)
        ys.append(norm.cdf(kappa * feat + bias).mean())
    return fit["grid_s"], np.array(ys)


def predict_accuracy_curve(fit, hi):
    k, sigma_hi, sigma_lo = fit["k"], fit["sigma_hi"], fit["sigma_lo"]
    sigma_c = sigma_hi if hi == 1 else sigma_lo
    bias_c = fit["bias_hi"] if hi == 1 else fit["bias_lo"]
    gs = fit["grid_s"]
    return gs, norm.cdf(k * gs / sigma_c + bias_c)


if __name__ == "__main__":
    import sys, pickle
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv"
    d = load(csv_path)
    fits = {}
    for animal in ("Ziggy", "Friedrich"):
        print(f"Fitting {animal}...")
        fits[animal] = fit_animal(d[d.animal == animal])
        print(f"  accuracy: k={fits[animal]['k']:.3f}  sigma_hi=1.000  "
              f"sigma_lo={fits[animal]['sigma_lo']:.3f}  bias_hi={fits[animal]['bias_hi']:+.3f}  "
              f"bias_lo={fits[animal]['bias_lo']:+.3f}")
        for m in E_MODELS:
            cf = fits[animal]["conf_fits"][m]
            print(f"  {m:17s}: kappa={cf['kappa']:+.3f}  bias={cf['bias']:+.3f}  nll={cf['nll']:.1f}")
        ee = fits[animal]["conf_fits"]["Entropy(e)"]
        print(f"  {'Entropy(e)':17s}: kappa={ee['kappa']:+.3f}  bias={ee['bias']:+.3f}  nll={ee['nll']:.1f}")
        ca = fits[animal]["conf_fits"]["CASANDRE"]
        print(f"  {'CASANDRE':17s}: kappa={ca['kappa']:+.3f}  bias={ca['bias']:+.3f}  "
              f"tau={ca['tau']:.3f}  nll={ca['nll']:.1f}")
        be = fits[animal]["conf_fits"]["Bayesian(|e|)"]
        print(f"  {'Bayesian(|e|)':17s}: kappa=+1.000 (fixed)  bias=+0.000 (fixed)  nll={be['nll']:.1f}")
        bc = fits[animal]["conf_fits"]["Bayesian(crit)"]
        print(f"  {'Bayesian(crit)':17s}: kappa=+1.000 (fixed)  bias={bc['bias']:+.3f}  nll={bc['nll']:.1f}")
        pg = fits[animal]["conf_fits"]["PositiveEvidence/Gain"]
        print(f"  {'PositiveEvidence/Gain':17s}: Q={pg['Q']:+.3f}  bias_hi={pg['bias_hi']:+.3f}  "
              f"bias_lo={pg['bias_lo']:+.3f}  nll={pg['nll']:.1f}")
        fb = fits[animal]["conf_fits"]["FullBayes(e)"]
        print(f"  {'FullBayes(e)':17s}: (zero free params, evaluated only)  nll={fb['nll']:.1f}")
        fbc = fits[animal]["conf_fits"]["FullBayes(crit)"]
        print(f"  {'FullBayes(crit)':17s}: bias={fbc['bias']:+.3f}  nll={fbc['nll']:.1f}")

    print()
    print("=" * 78)
    print("MODEL COMPARISON (AIC/BIC, accounts for differing free-parameter counts)")
    print("=" * 78)
    comparison_rows = []
    for animal in ("Ziggy", "Friedrich"):
        n_trials = int((d.animal == animal).sum())
        print(f"\n{animal} (n_trials={n_trials}):")
        rows = []
        for m in ALL_MODELS:
            nll = fits[animal]["conf_fits"][m]["nll"]
            k = MODEL_NPARAMS[m]
            a = aic(nll, k)
            b = bic(nll, k, n_trials)
            rows.append(dict(animal=animal, model=m, nparams=k, nll=nll, aic=a, bic=b))
        rows.sort(key=lambda r: r["bic"])
        best_aic = min(r["aic"] for r in rows)
        best_bic = min(r["bic"] for r in rows)
        print(f"  {'model':24s} {'k':>2s} {'nll':>10s} {'AIC':>12s} {'dAIC':>9s} "
              f"{'BIC':>12s} {'dBIC':>9s}")
        for r in rows:
            print(f"  {r['model']:24s} {r['nparams']:2d} {r['nll']:10.1f} "
                  f"{r['aic']:12.1f} {r['aic']-best_aic:+9.1f} "
                  f"{r['bic']:12.1f} {r['bic']-best_bic:+9.1f}")
        comparison_rows.extend(rows)
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("model_comparison_aic_bic.csv", index=False)
    print("\nSaved model_comparison_aic_bic.csv")

    with open("model_fits.pkl", "wb") as f:
        pickle.dump(fits, f)
    print("Saved model_fits.pkl")
