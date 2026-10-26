#!/usr/bin/env python3
"""
Figure 7: PositiveEvidence/Gain behavioral fit (Results,
sec:behavioral-model-comparison) -- the best-fitting model by both AIC and
BIC in both animals. Renamed from an earlier "PartialPositiveEvidence"
4-parameter fit (kappa, bias, g_lo, alpha): those 4 nominal parameters are
only identified up to 3 quantities (a shared |e|-slope and two
per-condition intercepts), so the fit here is the honest, directly-fit
3-parameter model (Q, bias_hi, bias_lo; see fit_free_param_models.py
module docstring item (7) and Supplemental Information). Same 8-panel
format (2 rows x 4 columns) as plot_model_fits.py's per-model figures and
as used for Figs. S1/S2; this is a thin wrapper that reuses
plot_model_fits.make_model_figure directly (with the "Model: X" suptitle
suppressed, since this is a main-text figure naming the model in its own
caption) so Fig. 7 stays pixel-for-pixel consistent with
figures/modelfit_PositiveEvidence_Gain.pdf.
"""
import pandas as pd
import fit_free_param_models as M
import plot_model_fits as P

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/real_behavior_full.csv"
    d = M.load(csv_path)
    fits = {animal: M.fit_animal(d[d.animal == animal]) for animal in ("Ziggy", "Friedrich")}
    _, bin_edges = pd.qcut(d["absori"], 8, duplicates="drop", retbins=True)
    P.make_model_figure(fits, d, "PositiveEvidence/Gain",
                         "figures/figure7_partial_pos_fit", bin_edges, suptitle=False)
