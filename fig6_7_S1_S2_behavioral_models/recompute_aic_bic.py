#!/usr/bin/env python3
"""
Recompute AIC/BIC for the confidence-model comparison (Methods,
model_comparison_aic_bic.csv), correcting for the fact that every fitted
model's reported "nll" in fit_free_param_models.py is actually the
L2-penalized optimization objective (raw nll + a small ridge on its
parameters), not the pure negative log-likelihood. This script re-derives
the pure nll for each model by subtracting the closed-form penalty term
evaluated at the fitted optimum, and reports AIC/BIC from that pure nll
instead.

(PositiveEvidence/Gain -- formerly a 4-nominal-parameter
"PartialPositiveEvidence" model, kappa/bias/g_lo/alpha -- is now fit and
reported directly as the 3 parameters the confidence likelihood actually
identifies, Q/bias_hi/bias_lo; see fit_free_param_models.py module
docstring item (7) and the Supplemental Information. There is no longer a
nominal-vs-identifiable parameter-count ambiguity to check here.)

Fits themselves are NOT changed (same optimizer, same ridge-stabilized
objective, same fixed RNG seeds) -- only the likelihood used for reporting
AIC/BIC is corrected, so this reproduces the original table's parameter
estimates exactly and only changes the nll/AIC/BIC columns.
"""
import numpy as np
import pandas as pd

import fit_free_param_models as M

L2 = M.L2_PENALTY


def penalty_for(model, cf):
    """Closed-form ridge penalty baked into fit_free_param_models.py's
    reported 'nll' for this model, evaluated at its fitted optimum."""
    if model in ("Calibrated", "ContrastBlind", "Entropy(e)"):
        return L2 * (cf["kappa"] ** 2 + cf["bias"] ** 2)
    if model == "CASANDRE":
        return L2 * (cf["kappa"] ** 2 + cf["bias"] ** 2) + 1e-3 * cf["tau"] ** 2
    if model == "PositiveEvidence/Gain":
        return L2 * (cf["Q"] ** 2 + cf["bias_hi"] ** 2 + cf["bias_lo"] ** 2)
    # Bayesian(|e|), Bayesian(crit), FullBayes(e), FullBayes(crit): unpenalized
    return 0.0


def main(csv_path="data/real_behavior_full.csv"):
    d = M.load(csv_path)
    fits = {}
    for animal in ("Ziggy", "Friedrich"):
        print(f"Fitting {animal}...")
        fits[animal] = M.fit_animal(d[d.animal == animal])

    rows = []
    for animal in ("Ziggy", "Friedrich"):
        n_trials = int((d.animal == animal).sum())
        for model in M.ALL_MODELS:
            cf = fits[animal]["conf_fits"][model]
            pen = penalty_for(model, cf)
            nll_reported = cf["nll"]
            nll_pure = nll_reported - pen
            k_nominal = M.MODEL_NPARAMS[model]
            row = dict(
                animal=animal, model=model,
                nparams_nominal=k_nominal,
                nll_reported_penalized=nll_reported,
                penalty=pen,
                nll_pure=nll_pure,
                aic_pure=M.aic(nll_pure, k_nominal),
                bic_pure=M.bic(nll_pure, k_nominal, n_trials),
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("model_comparison_aic_bic_corrected.csv", index=False)

    for animal in ("Ziggy", "Friedrich"):
        sub = df[df.animal == animal].sort_values("bic_pure")
        print(f"\n{animal} -- ranked by corrected (unpenalized-nll) BIC:")
        print(sub[["model", "nparams_nominal", "penalty", "nll_pure",
                    "aic_pure", "bic_pure"]].to_string(index=False))
        best_model_pure = sub.iloc[0]["model"]
        print(f"  Best model by corrected BIC: {best_model_pure}")

    print("\nSaved model_comparison_aic_bic_corrected.csv")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv")
