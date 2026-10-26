#!/usr/bin/env python3
"""
Model-recovery / confusion-matrix analysis for the 7 published confidence
models (Methods, sec:behavfit-methods). Exploratory validation only -- NOT
wired into the manuscript or any figure.

For each of the 7 published models used as a GENERATING model (using its
own real-data fitted parameters, for one animal, as ground truth) and for
each of --n_replicates independently simulated datasets (matching that
animal's real trial design exactly: same per-trial |orientation| and
contrast condition, same number of trials as the real session), this
script:

  1. Draws a per-trial signed decision variable e ~ Normal(k*s, sigma_c)
     (sigma_c = sigma_hi or sigma_lo by contrast condition), which exactly
     reproduces the shared accuracy model P(correct) = Phi(k*s/sigma_c):
     correct := (e > 0). This is mathematically identical to the two-step
     "sample correct, then sample |e| from the outcome-truncated normal"
     process fit_free_param_models.mc_abs_e_cache uses for its per-cell MC
     cache (that two-step form is just a caching device for aggregated
     cells; drawing e directly is simpler and exact for a per-trial
     simulator).
  2. Simulates a binary high/low confidence report from the GENERATING
     model's own equation, Bernoulli(p_conf), using each trial's own
     realized e / |e| -- not a cell-level average.
  3. Refits all 9 fit_free_param_models.ALL_MODELS variants to the
     synthetic (absori, hi, correct, confidence) dataset by calling
     fit_free_param_models.fit_animal() directly, so the refitting step is
     byte-for-byte the same code used on the real data (including its own
     fresh accuracy refit on the synthetic correct/incorrect outcomes --
     the recovery exercise does not get to peek at the true accuracy
     parameters during refitting).
  4. Picks the winning model (by AIC and by BIC, independently) among the
     7 PUBLISHED model variants only, dropping the two zero-free-parameter
     "ideal observer" variants from contention -- matching
     make_supp_fig_asym_criteria.py's own DROP_CODE_MODELS convention,
     since those two are not part of the manuscript's reported comparison.

Tabulates a (generating model) x (recovered/winning model) confusion
matrix, separately for AIC and BIC, and saves the raw per-replicate
results to CSV (one row per generating-model x replicate).

Usage:
    python3 model_recovery.py --animal Ziggy --n_replicates 20 --workers 8

Runs in parallel across (generating model, replicate) pairs via
ProcessPoolExecutor. Safe to interrupt and resume: already-completed
(gen_model, replicate) pairs found in the output CSV are skipped.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import fit_free_param_models as M

HERE = Path(__file__).resolve().parent

PUBLISHED_CODE_NAMES = [
    "Calibrated", "Bayesian(crit)", "ContrastBlind", "Entropy(e)",
    "CASANDRE", "FullBayes(crit)", "PositiveEvidence/Gain",
]
DISPLAY_NAME = {
    "Calibrated": "Calibrated",
    "Bayesian(crit)": "Bayesian(|e|)",
    "ContrastBlind": "ContrastBlind",
    "Entropy(e)": "Entropy(e)",
    "CASANDRE": "CASANDRE",
    "FullBayes(crit)": "FullBayes(e)",
    "PositiveEvidence/Gain": "PositiveEvidence/Gain",
}


def simulate_confidence(code_model, params, e_abs, e_signed, hi, acc, rng):
    """Vectorized per-trial P(high confidence) for one of the 7 published
    confidence models, then a Bernoulli draw. `acc` carries k, sigma_hi,
    sigma_lo, sigma_pooled, fullbayes_grid (the grid is only needed for
    Entropy(e)/FullBayes(crit))."""
    sigma_c = np.where(hi == 1, acc["sigma_hi"], acc["sigma_lo"])

    if code_model in ("Calibrated", "Bayesian(crit)"):
        kappa = 1.0 if code_model == "Bayesian(crit)" else params["kappa"]
        bias = params["bias"]
        feat = e_abs / sigma_c
        p = norm.cdf(kappa * feat + bias)
    elif code_model == "ContrastBlind":
        feat = e_abs / acc["sigma_pooled"]
        p = norm.cdf(params["kappa"] * feat + params["bias"])
    elif code_model == "Entropy(e)":
        feat = M.fullbayes_e_neg_entropy(e_signed, acc["fullbayes_grid"])
        p = norm.cdf(params["kappa"] * feat + params["bias"])
    elif code_model == "CASANDRE":
        z = rng.standard_normal(len(e_abs))
        sigma_hat = sigma_c * np.exp(params["tau"] * z)
        feat = e_abs / sigma_hat
        p = norm.cdf(params["kappa"] * feat + params["bias"])
    elif code_model == "FullBayes(crit)":
        p_hyp = np.clip(M.fullbayes_e_readout(e_signed, acc["fullbayes_grid"]), M.EPS, 1 - M.EPS)
        z = norm.ppf(p_hyp) + params["bias"]
        p = norm.cdf(z)
    elif code_model == "PositiveEvidence/Gain":
        bias_c = np.where(hi == 1, params["bias_hi"], params["bias_lo"])
        p = norm.cdf(params["Q"] * e_abs + bias_c)
    else:
        raise ValueError(code_model)

    p = np.clip(p, 1e-12, 1 - 1e-12)
    conf = (rng.random(len(p)) < p).astype(int)
    return conf


def simulate_dataset(code_model, params, s, hi, acc, rng):
    """s, hi: the real per-trial design (absori, contrast condition),
    reused exactly so the synthetic dataset has the same design as the
    real session. Returns a DataFrame with absori/hi/correct/confidence,
    ready to feed straight into fit_free_param_models.fit_animal()."""
    sigma_c = np.where(hi == 1, acc["sigma_hi"], acc["sigma_lo"])
    mean = acc["k"] * s
    e = rng.normal(mean, sigma_c)
    correct = (e > 0).astype(int)
    e_abs = np.abs(e)
    # e_signed convention used throughout fit_free_param_models: +|e| if
    # correct, -|e| if error -- which is just e itself, since correct<=>e>0.
    confidence = simulate_confidence(code_model, params, e_abs, e, hi, acc, rng)
    return pd.DataFrame(dict(absori=s, hi=hi, correct=correct, confidence=confidence))


def ground_truth_params(code_model, conf_fits):
    cf = conf_fits[code_model]
    if code_model in ("Calibrated", "ContrastBlind", "Entropy(e)"):
        return dict(kappa=cf["kappa"], bias=cf["bias"])
    if code_model == "Bayesian(crit)":
        return dict(bias=cf["bias"])
    if code_model == "CASANDRE":
        return dict(kappa=cf["kappa"], bias=cf["bias"], tau=cf["tau"])
    if code_model == "FullBayes(crit)":
        return dict(bias=cf["bias"])
    if code_model == "PositiveEvidence/Gain":
        return dict(Q=cf["Q"], bias_hi=cf["bias_hi"], bias_lo=cf["bias_lo"])
    raise ValueError(code_model)


def winner(fit, n_trials, criterion):
    best_model, best_val = None, np.inf
    for m in PUBLISHED_CODE_NAMES:
        nll = fit["conf_fits"][m]["nll"]
        k = M.MODEL_NPARAMS[m]
        val = M.aic(nll, k) if criterion == "aic" else M.bic(nll, k, n_trials)
        if val < best_val:
            best_val, best_model = val, m
    return best_model


def _run_one(task):
    """Top-level (picklable) worker: simulate one (gen_model, replicate)
    dataset and refit all models. task is a plain dict so this works
    cleanly under ProcessPoolExecutor."""
    gen_model = task["gen_model"]
    params = task["params"]
    s = task["s"]
    hi = task["hi"]
    acc = task["acc"]
    seed = task["seed"]
    rep = task["rep"]
    animal = task["animal"]

    rng = np.random.default_rng(seed)
    t0 = time.time()
    sim_df = simulate_dataset(gen_model, params, s, hi, acc, rng)
    fit = M.fit_animal(sim_df)
    n_trials = len(sim_df)
    win_aic = winner(fit, n_trials, "aic")
    win_bic = winner(fit, n_trials, "bic")
    dt = time.time() - t0
    return dict(animal=animal, gen_model=gen_model, gen_display=DISPLAY_NAME[gen_model],
                replicate=rep, win_aic=win_aic, win_aic_display=DISPLAY_NAME[win_aic],
                win_bic=win_bic, win_bic_display=DISPLAY_NAME[win_bic], seconds=dt)


def load_done(out_csv):
    p = Path(out_csv)
    if not p.exists():
        return set(), []
    df = pd.read_csv(p)
    done = set(zip(df["gen_model"], df["replicate"]))
    return done, df.to_dict("records")


def run(animal, n_replicates, seed, out_csv, workers, limit_models=None):
    d = M.load(str(HERE / "data" / "real_behavior_full.csv"))
    d_animal = d[d.animal == animal].reset_index(drop=True)
    s = d_animal["absori"].to_numpy()
    hi = d_animal["hi"].to_numpy()

    with open(HERE / "model_fits.pkl", "rb") as f:
        real_fits = pickle.load(f)
    real_fit = real_fits[animal]
    acc = dict(k=real_fit["k"], sigma_hi=real_fit["sigma_hi"], sigma_lo=real_fit["sigma_lo"],
               sigma_pooled=real_fit["sigma_pooled"], fullbayes_grid=real_fit["fullbayes_grid"])

    models = limit_models or PUBLISHED_CODE_NAMES
    done, rows = load_done(out_csv)
    if done:
        print(f"Resuming: {len(done)} (gen_model, replicate) pairs already done in {out_csv}")

    tasks = []
    for gen_model in models:
        params = ground_truth_params(gen_model, real_fit["conf_fits"])
        for rep in range(n_replicates):
            if (gen_model, rep) in done:
                continue
            tasks.append(dict(gen_model=gen_model, params=params, s=s, hi=hi, acc=acc,
                               seed=seed * 100003 + hash((gen_model, rep)) % 1_000_000,
                               rep=rep, animal=animal))

    total = len(models) * n_replicates
    t0 = time.time()
    n_completed = len(rows)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, t): t for t in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                rows.append(r)
                n_completed += 1
                elapsed = time.time() - t0
                rate = elapsed / max(1, n_completed - len(done))
                remaining = (total - n_completed) * rate
                print(f"[{n_completed}/{total}] gen={r['gen_model']:22s} rep={r['replicate']:3d} "
                      f"-> AIC winner={r['win_aic']:22s} BIC winner={r['win_bic']:22s} "
                      f"({r['seconds']:.1f}s this task, ~{remaining/60:.1f} min remaining)",
                      flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def confusion_matrix(results_df, criterion, models=None):
    models = models or PUBLISHED_CODE_NAMES
    col = f"win_{criterion}_display"
    models_disp = [DISPLAY_NAME[m] for m in models]
    cm = pd.crosstab(results_df["gen_display"], results_df[col])
    cm = cm.reindex(index=models_disp, columns=models_disp, fill_value=0)
    return cm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="Ziggy", choices=["Ziggy", "Friedrich"])
    ap.add_argument("--n_replicates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None,
                     help="comma-separated subset of PUBLISHED_CODE_NAMES, for quick tests")
    args = ap.parse_args()
    out_csv = args.out or str(HERE / f"model_recovery_{args.animal}.csv")
    limit_models = args.models.split(",") if args.models else None
    df = run(args.animal, args.n_replicates, args.seed, out_csv, args.workers, limit_models)
    print(f"\nSaved {out_csv} ({len(df)} rows)")
    print("\n=== AIC confusion matrix (rows=generating, cols=recovered) ===")
    print(confusion_matrix(df, "aic", limit_models))
    print("\n=== BIC confusion matrix (rows=generating, cols=recovered) ===")
    print(confusion_matrix(df, "bic", limit_models))
