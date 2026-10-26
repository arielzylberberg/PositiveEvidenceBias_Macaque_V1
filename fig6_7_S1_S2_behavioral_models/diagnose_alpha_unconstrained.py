#!/usr/bin/env python3
"""
Reproduces, standalone, the identifiability proof behind the
PositiveEvidence/Gain model (fit_free_param_models.py module docstring
item (7); Supplemental Information, sec:alpha-supp): the chosen/unchosen
pool-weighting family's 4 nominal parameters (kappa, bias, g_lo, alpha)
are only identified up to 3 quantities (a shared |e|-slope and two
per-condition intercepts), so fitting all 4 -- with alpha either fully
unconstrained (alpha = 1 - exp(theta_alpha), ranging over (-inf, 1]) or
bounded to (-1,1) via tanh -- reproduces the same log-likelihood as the
direct, explicitly-3-parameter fit that fit_free_param_models.py now
reports as "PositiveEvidence/Gain" (Q, bias_hi, bias_lo). The two
4-parameter fitting functions here (neg_ll_partial_positive_unconstrained/
fit_unconstrained for the unbounded transform, neg_ll_partial_positive_
constrained/fit_constrained for the tanh-bounded one) are frozen,
self-contained copies of an earlier, retired parametrization -- kept here,
decoupled from the live module, purely to keep this identifiability check
independently reproducible.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import fit_free_param_models as M

EPS = M.EPS


def neg_ll_partial_positive_unconstrained(theta, cache, n_cell_trials, k_conf_sum):
    kappa, bias, log_g_lo, theta_alpha = theta
    g_lo = np.exp(log_g_lo)
    alpha = 1.0 - np.exp(theta_alpha)  # unbounded above at 1, unbounded below
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        g_c = 1.0 if hi == 1 else g_lo
        feat = partial_pos_feature(cache[cell], g_c, alpha)
        p_bar = np.clip(norm.cdf(kappa * feat + bias).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    return total  # no ridge -- this is the pre-ridge, pre-bounding version


def fit_unconstrained(cache, n_cell_trials, k_conf_sum):
    best = None
    for theta_alpha0 in (0.0, 1.0, -1.0, 2.0, -2.0):
        res = minimize(neg_ll_partial_positive_unconstrained, x0=[1.0, 0.0, 0.0, theta_alpha0],
                        args=(cache, n_cell_trials, k_conf_sum),
                        method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=8000))
        if best is None or res.fun < best.fun:
            best = res
    kappa, bias, log_g_lo, theta_alpha = best.x
    g_lo = np.exp(log_g_lo)
    alpha = 1.0 - np.exp(theta_alpha)
    return dict(kappa=kappa, bias=bias, g_lo=g_lo, alpha=alpha, nll=best.fun, ok=best.success)


def partial_pos_feature(e_abs, g_c, alpha):
    """chosen - alpha*unchosen; see fit_free_param_models.py module
    docstring item (7) for the derivation. Frozen local copy -- the live
    module no longer defines this (see this file's module docstring)."""
    return g_c * (1.0 - alpha) / 2.0 + e_abs * (1.0 + alpha) / 2.0


def neg_ll_partial_positive_constrained(theta, cache, n_cell_trials, k_conf_sum):
    kappa, bias, log_g_lo, theta_alpha = theta
    g_lo = np.exp(log_g_lo)
    alpha = np.tanh(theta_alpha)
    total = 0.0
    for cell, n_tr in n_cell_trials.items():
        if n_tr == 0:
            continue
        s_val, hi, outcome = cell
        g_c = 1.0 if hi == 1 else g_lo
        feat = partial_pos_feature(cache[cell], g_c, alpha)
        p_bar = np.clip(norm.cdf(kappa * feat + bias).mean(), EPS, 1 - EPS)
        k_conf = k_conf_sum[cell]
        total += -(k_conf * np.log(p_bar) + (n_tr - k_conf) * np.log(1 - p_bar))
    ridge = 2e-3 * (kappa ** 2 + bias ** 2) + 2e-3 * (theta_alpha ** 2 + log_g_lo ** 2)
    return total + ridge


def fit_constrained(cache, n_cell_trials, k_conf_sum):
    best = None
    for theta_alpha0 in (0.0, 1.0, -1.0, 2.0, -2.0):
        res = minimize(neg_ll_partial_positive_constrained, x0=[1.0, 0.0, 0.0, theta_alpha0],
                        args=(cache, n_cell_trials, k_conf_sum),
                        method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=6000))
        if best is None or res.fun < best.fun:
            best = res
    kappa, bias, log_g_lo, theta_alpha = best.x
    g_lo = np.exp(log_g_lo)
    alpha = np.tanh(theta_alpha)
    return dict(kappa=kappa, bias=bias, g_hi=1.0, g_lo=g_lo, alpha=alpha,
                nll=best.fun, ok=best.success)


def neg_ll_reduced3(theta, cache, n_cell_trials, k_conf_sum):
    """Phi(Q*|e| + bias_hi_or_lo): shared slope Q on |e|, free per-condition intercept."""
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
    return total


def fit_reduced3(cache, n_cell_trials, k_conf_sum):
    best = None
    for x0 in ([1.0, 0.0, 0.0], [0.5, -0.5, -0.5], [2.0, 0.5, 0.5]):
        res = minimize(neg_ll_reduced3, x0=x0, args=(cache, n_cell_trials, k_conf_sum),
                        method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-8, maxiter=8000))
        if best is None or res.fun < best.fun:
            best = res
    return dict(Q=best.x[0], bias_hi=best.x[1], bias_lo=best.x[2], nll=best.fun, ok=best.success)


def cells_for_animal(d_animal, n_sbins=40):
    d = M.prepare_cells(d_animal, n_sbins=n_sbins)
    s = d["s_center"].to_numpy()
    hi = d["hi"].to_numpy()
    correct = d["correct"].to_numpy()
    conf = d["confidence"].to_numpy()
    acc_fit = M.fit_accuracy(s, hi, correct)
    k, sigma_hi, sigma_lo, bias_hi, bias_lo = (acc_fit["k"], acc_fit["sigma_hi"],
                                                acc_fit["sigma_lo"], acc_fit["bias_hi"],
                                                acc_fit["bias_lo"])
    grid_s = np.linspace(np.nanmin(s), np.nanmax(s), 60)
    unique_present = set(zip(np.round(s, 6), hi.astype(int), correct.astype(int)))
    grid_cells = set((float(np.round(gs, 6)), h, o) for gs in grid_s for h in (0, 1) for o in (0, 1))
    all_cells = unique_present | grid_cells
    cache = M.mc_abs_e_cache(all_cells, k, sigma_hi, sigma_lo, bias_hi=bias_hi, bias_lo=bias_lo)
    cell_key = list(zip(np.round(s, 6), hi.astype(int), correct.astype(int)))
    df_cells = pd.DataFrame({"cell": cell_key, "conf": conf})
    grp = df_cells.groupby("cell")["conf"].agg(["sum", "count"])
    n_cell_trials = grp["count"].to_dict()
    k_conf_sum = grp["sum"].to_dict()
    return cache, n_cell_trials, k_conf_sum


def main(csv_path="data/real_behavior_full.csv"):
    d = M.load(csv_path)
    for animal in ("Ziggy", "Friedrich"):
        cache, n_cell_trials, k_conf_sum = cells_for_animal(d[d.animal == animal])
        constrained = fit_constrained(cache, n_cell_trials, k_conf_sum)
        unconstrained = fit_unconstrained(cache, n_cell_trials, k_conf_sum)
        reduced3 = fit_reduced3(cache, n_cell_trials, k_conf_sum)
        # constrained nll includes the ridge penalty; strip it for a fair nll comparison
        theta_alpha_c = np.arctanh(np.clip(constrained["alpha"], -1 + 1e-12, 1 - 1e-12))
        log_g_lo_c = np.log(constrained["g_lo"])
        pen_c = M.L2_PENALTY * (constrained["kappa"] ** 2 + constrained["bias"] ** 2) \
            + 2e-3 * (theta_alpha_c ** 2 + log_g_lo_c ** 2)  # matches fit_constrained's ridge
        constrained_pure_nll = constrained["nll"] - pen_c
        print(f"\n=== {animal} ===")
        print(f"  constrained (tanh, bounded (-1,1)):   alpha={constrained['alpha']:+.4f}  "
              f"kappa={constrained['kappa']:+.4f}  g_lo={constrained['g_lo']:.4f}  "
              f"pure_nll={constrained_pure_nll:.4f}")
        print(f"  unconstrained (1-exp, unbounded):     alpha={unconstrained['alpha']:+.4f}  "
              f"kappa={unconstrained['kappa']:+.4f}  g_lo={unconstrained['g_lo']:.4f}  "
              f"nll={unconstrained['nll']:.4f}")
        print(f"  nll improvement (unconstrained vs constrained): "
              f"{constrained_pure_nll - unconstrained['nll']:.4f} nats")
        print(f"  reduced 3-param (shared Q, free bias_hi/bias_lo): "
              f"Q={reduced3['Q']:+.4f}  bias_hi={reduced3['bias_hi']:+.4f}  "
              f"bias_lo={reduced3['bias_lo']:+.4f}  nll={reduced3['nll']:.4f}")
        print(f"  |unconstrained.nll - reduced3.nll| = "
              f"{abs(unconstrained['nll'] - reduced3['nll']):.4f} nats")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv")
