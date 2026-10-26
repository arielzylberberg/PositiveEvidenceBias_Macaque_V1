"""
stats_summary.py
==================

Implements reviewer comment #1 (comments.pdf, page 1):

    "We could perform a statistical test comparing the mean activity
    differences across splits. Specifically, compare the chosen-pref
    activity between the confidence split and the accuracy split.
    Ideally, there should be no significant difference. Then repeat the
    same analysis for the unchosen-pref activity, where we would ideally
    observe a significant difference. Both tests should be paired
    across sessions."

Also reproduces the within-panel bracket stat already shown in the
reference figure (chosen-pref vs unchosen-pref, paired across sessions,
separately for the confidence split and the accuracy split).

All tests are paired (each session contributes one value per condition)
and reported both as Wilcoxon signed-rank (matches the FDR-free,
non-parametric convention used throughout analysis_az/, e.g.
run_pref_npref_ratio.m's `signrank`, compare_ab_conf_vs_acc.py's
`stats.wilcoxon`) and as a paired t-test for reference.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def sig_str(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_report(label, a, b, a_name, b_name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    n = len(a)
    diff = a - b
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    print(f"\n--- {label}  (n = {n} sessions, paired) ---")
    print(f"  {a_name:<28s} mean = {a.mean():+.4f}   median = {np.median(a):+.4f}")
    print(f"  {b_name:<28s} mean = {b.mean():+.4f}   median = {np.median(b):+.4f}")
    print(f"  mean paired diff ({a_name} - {b_name}) = {diff.mean():+.4f}")
    print(f"  Wilcoxon signed-rank (two-sided): W={w.statistic:.1f}  p={w.pvalue:.4g}  {sig_str(w.pvalue)}")
    print(f"  Paired t-test        (two-sided): t={t.statistic:+.3f}  p={t.pvalue:.4g}  {sig_str(t.pvalue)}")
    print(f"  Cohen's dz (paired effect size) = {dz:+.3f}")
    return dict(label=label, n=n, mean_a=a.mean(), mean_b=b.mean(),
                mean_diff=diff.mean(), wilcoxon_p=w.pvalue, ttest_p=t.pvalue,
                cohens_dz=dz)


def main(csv_path: str, out_csv: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} sessions from {csv_path}\n")
    print("=" * 72)
    print("WITHIN-PANEL:  Chosen-pref vs Unchosen-pref  (the bracket already")
    print("shown in the reference figure)")
    print("=" * 72)
    r1 = paired_report("Confidence split: Chosen-pref vs Unchosen-pref",
                        df.chosen_conf_diff, df.unchosen_conf_diff,
                        "Chosen-pref (conf)", "Unchosen-pref (conf)")
    r2 = paired_report("Accuracy split: Chosen-pref vs Unchosen-pref",
                        df.chosen_acc_diff, df.unchosen_acc_diff,
                        "Chosen-pref (acc)", "Unchosen-pref (acc)")

    print("\n" + "=" * 72)
    print("CROSS-SPLIT (reviewer comment #1):  Confidence split vs Accuracy split")
    print("Hypothesis: chosen-pref should NOT differ; unchosen-pref SHOULD differ")
    print("=" * 72)
    r3 = paired_report("Chosen-pref: Confidence-split vs Accuracy-split",
                        df.chosen_conf_diff, df.chosen_acc_diff,
                        "Chosen-pref (conf)", "Chosen-pref (acc)")
    r4 = paired_report("Unchosen-pref: Confidence-split vs Accuracy-split",
                        df.unchosen_conf_diff, df.unchosen_acc_diff,
                        "Unchosen-pref (conf)", "Unchosen-pref (acc)")

    out = pd.DataFrame([r1, r2, r3, r4])
    out.to_csv(out_csv, index=False)
    print(f"\nSaved summary table -> {out_csv}")
    return out


if __name__ == "__main__":
    csv_path = os.path.join(HERE, "data", "figure1_activity.csv")
    out_csv = os.path.join(HERE, "data", "figure1_stats_summary.csv")
    main(csv_path, out_csv)
