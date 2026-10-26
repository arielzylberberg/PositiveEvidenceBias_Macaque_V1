#!/usr/bin/env python3
"""Why accuracy is not at chance where P(right choice) = 50%.

Supplement to Fig. 2. If d were the quantity the animal's choice is read out
from, then at d = 0 the choice would be uninformative about the stimulus and
accuracy would be 0.50. It is ~0.77. This figure shows that the gap is
measurement noise in d, quantitatively.

  A  Split-half reliability of d, computed across disjoint halves of each
     session's two preference pools (Spearman-Brown corrected).
  B  The measurement-noise model tested directly. With
     d = sqrt(rho) D* + noise, conditioning on d = 0 leaves
     E[D* | d = 0, stimulus] = mu (1 - rho) with posterior SD sqrt(1 - rho),
     so accuracy at d = 0 should be Phi(mu sqrt(1 - rho)) -- a curve that
     reaches chance only as rho reaches 1. Sessions with a more reliable d
     do sit lower, along that curve.
  C  Propagating the implied single-neuron reliability through
     Spearman-Brown: how many neurons would be needed for accuracy at
     d = 0 to approach chance.

All quantities use the same pooled axis d = E+ - E- as Fig. 2.
"""
import glob
import os
import warnings

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "session_cache")

import sys
sys.path.insert(0, os.path.join(HERE, "..", "shared"))
import pub_style as PS  # noqa: E402

# v2: font sizes and the accuracy colour come from the shared pub_style
# module (previously this script's own, independent font scale) -- see
# shared/pub_style.py. NEU is not a pub_style semantic colour (it labels
# a single "number of neurons" series that appears nowhere else in the
# paper), so it is kept as a local accent colour rather than moved into
# the shared palette.
FS_LBL, FS_TICK, FS_LEG, FS_TTL, FS_PANEL = (
    PS.FS_LBL, PS.FS_TICK, PS.FS_LEG, PS.FS_TTL, PS.FS_PANEL)
ACC = PS.ACCR_CORR
NEU = "#3d5a80"
PS.apply_rcparams()

SLICE = 0.25          # |d| < SLICE SD counts as "near d = 0"
FRACS = [0.25, 0.4, 0.6, 0.8, 1.0]

rel, obs_acc, pred_acc, npop, mu_all = [], [], [], [], []
frac_acc = {f: [] for f in FRACS}
frac_n = {f: [] for f in FRACS}

for f in sorted(glob.glob(os.path.join(CACHE, "session_*.npz"))):
    z = np.load(f)
    if z["right_pref"].sum() < 3 or z["left_pref"].sum() < 3:
        continue
    C = z["counts"].astype(float)
    ch = (z["choice"] == 1).astype(int)
    ac = z["correct"].astype(int)
    mu_, sd = C.mean(0), C.std(0)
    keep = sd > 0
    Z = (C[:, keep] - mu_[keep]) / sd[keep]
    ri, li = np.where(z["right_pref"][keep])[0], np.where(z["left_pref"][keep])[0]
    rng = np.random.default_rng(0)

    def dz_from(rsub, lsub):
        d = Z[:, rsub].mean(1) - Z[:, lsub].mean(1)
        return (d - d.mean()) / d.std()

    # --- A: split-half reliability -----------------------------------
    if len(ri) >= 4 and len(li) >= 4:
        r = []
        for _ in range(30):
            pr, pl = rng.permutation(ri), rng.permutation(li)
            hr, hl = len(ri) // 2, len(li) // 2
            a = Z[:, pr[:hr]].mean(1) - Z[:, pl[:hl]].mean(1)
            b = Z[:, pr[hr:2*hr]].mean(1) - Z[:, pl[hl:2*hl]].mean(1)
            r.append(np.corrcoef(a, b)[0, 1])
        rh = float(np.mean(r))
        rho = 2 * rh / (1 + rh)
    else:
        continue
    rel.append(rho)
    npop.append(len(ri) + len(li))

    # --- B: accuracy near d = 0 vs fraction of neurons used ----------
    for fr in FRACS:
        nr, nl = max(2, int(round(fr * len(ri)))), max(2, int(round(fr * len(li))))
        vals = []
        for _ in range(1 if fr == 1.0 else 12):
            rs = ri if fr == 1.0 else rng.choice(ri, nr, replace=False)
            ls = li if fr == 1.0 else rng.choice(li, nl, replace=False)
            dz = dz_from(rs, ls)
            m = np.abs(dz) < SLICE
            if m.sum() >= 30:
                vals.append(ac[m].mean())
        if vals:
            frac_acc[fr].append(np.mean(vals))
            frac_n[fr].append(nr + nl)

    # --- C: model prediction vs observation --------------------------
    dz = dz_from(ri, li)
    m = np.abs(dz) < SLICE
    if m.sum() >= 30:
        mu = norm.ppf(min(ac.mean(), 0.999))
        mu_all.append(mu)
        pred_acc.append(norm.cdf(mu * np.sqrt(max(1 - rho, 0))))
        obs_acc.append(ac[m].mean())

rel = np.array(rel)
pred_acc, obs_acc = np.array(pred_acc), np.array(obs_acc)

fig = plt.figure(figsize=(10.4, 3.7))
gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.185, top=0.855,
                      wspace=0.42)
axA, axB, axD = (fig.add_subplot(gs[0, i]) for i in range(3))

# A -----------------------------------------------------------------
axA.hist(rel, bins=np.arange(0.2, 1.0, 0.08), facecolor=NEU, alpha=0.75,
         edgecolor="0.15", linewidth=0.6)
axA.axvline(rel.mean(), color="0.1", ls="--", lw=1.6)
axA.axvline(1.0, color="0.5", ls=":", lw=1.0)
axA.set_xlabel(r"split-half reliability of $d$", fontsize=FS_LBL)
axA.set_ylabel("sessions", fontsize=FS_LBL)
axA.set_title(f"mean $\\rho$ = {rel.mean():.2f}", fontsize=FS_TTL, pad=7)
axA.spines[["top", "right"]].set_visible(False)

# B -----------------------------------------------------------------
rr = np.linspace(0.05, 0.999, 200)
mu_m = float(np.mean(mu_all))
axB.plot(rr, norm.cdf(mu_m * np.sqrt(1 - rr)), color="0.35", lw=1.8,
         label=r"$\Phi(\mu\sqrt{1-\rho})$")
axB.scatter(rel, obs_acc, s=40, facecolor=ACC, edgecolor="0.15",
            linewidth=0.5, alpha=0.85, zorder=4, label="sessions")
axB.axhline(0.5, color="0.35", ls="--", lw=1.0)
axB.set_xlim(0, 1.02); axB.set_ylim(0.45, 0.95)
axB.set_xlabel(r"reliability of $d$,  $\rho$", fontsize=FS_LBL)
axB.set_ylabel(f"accuracy at $|d|$ < {SLICE} SD", fontsize=FS_LBL)
axB.set_title(f"model vs data  (r = {np.corrcoef(rel, obs_acc)[0,1]:+.2f})",
              fontsize=FS_TTL, pad=7)
axB.legend(frameon=False, loc="lower left")
axB.spines[["top", "right"]].set_visible(False)

# D -----------------------------------------------------------------
rho1 = np.mean([r / (n - (n - 1) * r) for r, n in zip(rel, npop)])
mu = float(np.mean(mu_all))
nn = np.array([10, 20, 31, 60, 100, 200, 400, 1000, 3000, 10000, 30000])
rn = nn * rho1 / (1 + (nn - 1) * rho1)
axD.plot(nn, norm.cdf(mu * np.sqrt(np.clip(1 - rn, 0, 1))), "-", color=NEU, lw=2)
axD.axhline(0.5, color="0.35", ls="--", lw=1.0)
axD.scatter([np.mean(npop)], [obs_acc.mean()], marker="o", s=55,
            facecolor="white", edgecolor=ACC, linewidth=1.8, zorder=5)
axD.annotate("this study", (np.mean(npop), obs_acc.mean()),
             textcoords="offset points", xytext=(8, 6), fontsize=FS_LEG,
             color="0.25")
axD.set_xscale("log")
axD.set_ylim(0.45, 0.85)
axD.set_xlabel("neurons recorded (extrapolated)", fontsize=FS_LBL)
axD.set_ylabel("accuracy at $d$ = 0", fontsize=FS_LBL)
axD.set_title(f"per-neuron $\\rho$ = {rho1:.3f}", fontsize=FS_TTL, pad=7)
axD.spines[["top", "right"]].set_visible(False)

for ax, s in ((axA, "A"), (axB, "B"), (axD, "C")):
    ax.text(-0.26, 1.05, s, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left", clip_on=False)

FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "figS_d_reliability.pdf"))
fig.savefig(os.path.join(FIGDIR, "figS_d_reliability.png"), dpi=160)

print(f"{len(rel)} sessions with >=4 neurons per pool")
print(f"A  reliability rho: mean {rel.mean():.3f}, range {rel.min():.3f}-{rel.max():.3f}")
print(f"B  corr(rho, accuracy at d~0) across sessions = "
      f"{np.corrcoef(rel, obs_acc)[0,1]:+.3f}; "
      f"mean predicted {pred_acc.mean():.3f} vs observed {obs_acc.mean():.3f}")
print(f"C  implied per-neuron reliability {rho1:.4f}; "
      f"predicted accuracy at d = 0 for 400 neurons "
      f"{norm.cdf(mu*np.sqrt(1-400*rho1/(1+399*rho1))):.3f}, "
      f"for 6000 {norm.cdf(mu*np.sqrt(1-6000*rho1/(1+5999*rho1))):.3f}")
print("Saved figS_d_reliability.pdf")
