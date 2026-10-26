#!/usr/bin/env python3
"""Do the simple axes need replacing? Side-by-side with the decoder axes.

Both rows use the SAME improved machinery -- axes z-scored within session,
surfaces fitted per session as logistic tensor-product splines in (d, g),
then averaged across sessions, with coverage-based masking. The only thing
that differs is how d and g are defined:

  row 1  simple    d = mean(E+) - mean(E-),  g = mean(E+) + mean(E-)
                   over the significance-selected pools -- the published
                   definition, so E+/E- and the b = -a vs b = 0 geometry of
                   the pool-fit analysis survive intact.
  row 2  decoder   d = w.x from a cross-validated logistic choice decoder,
                   g = |w|.x orthogonalised against d.
  row 3  the difference, on a symmetric scale.

OUTPUT figures/fig2_axis_comparison.pdf / .png
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_compare import run, grid, LIM      # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import pub_style as PS  # noqa: E402

# v2: font sizes and colours now come from the shared pub_style module
# (previously this script's own, independent scale) -- see
# shared/pub_style.py. Colour values are unchanged from before.
FS_LBL, FS_TICK, FS_LEG, FS_TTL, FS_PANEL = (
    PS.FS_LBL, PS.FS_TICK, PS.FS_LEG, PS.FS_TTL, PS.FS_PANEL)
CHOI_RIGHT, CHOI_LEFT = PS.CHOI_RIGHT, PS.CHOI_LEFT
CONF_HIGH, CONF_LOW = PS.CONF_HIGH, PS.CONF_LOW
ACCR_CORR = PS.ACCR_CORR
MASK_FILL = "#e6e6e6"
PS.apply_rcparams()


def diverging(lo, hi):
    cm = LinearSegmentedColormap.from_list("d", [lo, "#ffffff", hi])
    cm.set_bad(MASK_FILL)
    return cm


def sequential(hi, floor=0.14):
    rgb = np.array(mpl.colors.to_rgb(hi))
    cm = LinearSegmentedColormap.from_list(
        "s", [floor * rgb + (1 - floor) * np.ones(3), rgb])
    cm.set_bad(MASK_FILL)
    return cm


simple, m_s, n = run("simple")
dec, m_d, _ = run("decoder_orth")
both = m_s & m_d

SPEC = (("choice", "Choice", diverging(CHOI_LEFT, CHOI_RIGHT), 0, 1,
         "P(right choice)"),
        ("conf", "Confidence", diverging(CONF_LOW, CONF_HIGH), 0.25, 0.75,
         "P(high confidence)"),
        ("acc", "Accuracy", sequential(ACCR_CORR), 0.77, 0.90, "P(correct)"))

fig = plt.figure(figsize=(11.4, 11.0))
gs = fig.add_gridspec(3, 3, left=0.10, right=0.94, bottom=0.055, top=0.90,
                      wspace=0.50, hspace=0.34)

for col, (key, title, cmap, vlo, vhi, cblab) in enumerate(SPEC):
    for row, (M, rowlab) in enumerate(((simple, "simple axes"),
                                       (dec, "decoder axes"))):
        ax = fig.add_subplot(gs[row, col])
        pc = ax.pcolormesh(grid, grid, np.ma.masked_array(M[key], ~both),
                           cmap=cmap, vmin=vlo, vmax=vhi, shading="nearest")
        ax.set_aspect("equal", adjustable="box")
        ax.axvline(0, color="0.3", ls=":", lw=0.8)
        ax.axhline(0, color="0.3", ls=":", lw=0.8)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        if row == 0:
            ax.set_title(title, fontsize=FS_TTL, pad=6)
        if col == 0:
            ax.set_ylabel(f"{rowlab}\n$g$ (SD)", fontsize=FS_LBL)
        cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cblab, fontsize=FS_TICK - 0.5)
        cb.ax.tick_params(labelsize=FS_TICK - 1)
        cb.outline.set_linewidth(0.6)

    ax = fig.add_subplot(gs[2, col])
    D = np.ma.masked_array(np.asarray(dec[key]) - np.asarray(simple[key]), ~both)
    lim = float(np.max(np.abs(D)))
    cmd = LinearSegmentedColormap.from_list("dd", ["#3b4cc0", "#ffffff", "#b40426"])
    cmd.set_bad(MASK_FILL)
    pc = ax.pcolormesh(grid, grid, D, cmap=cmd, vmin=-lim, vmax=lim,
                       shading="nearest")
    ax.set_aspect("equal", adjustable="box")
    ax.axvline(0, color="0.3", ls=":", lw=0.8)
    ax.axhline(0, color="0.3", ls=":", lw=0.8)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_xlabel("$d$ (SD)", fontsize=FS_LBL)
    if col == 0:
        ax.set_ylabel("decoder − simple\n$g$ (SD)", fontsize=FS_LBL)
    r = np.corrcoef(np.asarray(dec[key])[both], np.asarray(simple[key])[both])[0, 1]
    ax.set_title(f"r = {r:.3f},  max |Δ| = {lim:.3f}", fontsize=FS_TTL - 1, pad=5)
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=FS_TICK - 1)
    cb.outline.set_linewidth(0.6)

fig.legend(handles=[Patch(facecolor=MASK_FILL, edgecolor="#cccccc",
                          label="outside the coverage of both definitions")],
           loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False,
           fontsize=FS_LEG)
fig.suptitle("Same per-session surface pipeline, two definitions of the axes "
             f"({n} sessions)", fontsize=FS_TTL + 2, y=0.955)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig2_axis_comparison.pdf"))
fig.savefig(os.path.join(out, "fig2_axis_comparison.png"), dpi=150)


def at(M, dv, gv):
    return M[int(np.argmin(np.abs(grid - gv))), int(np.argmin(np.abs(grid - dv)))]


print(f"{n} sessions\n")
for name, M in (("simple ", simple), ("decoder", dec)):
    c0, c1 = at(M["conf"], 0, -1.5), at(M["conf"], 0, 1.5)
    a0, a1 = at(M["acc"], 0, -1.5), at(M["acc"], 0, 1.5)
    print(f"  {name}  at d=0, g -1.5->+1.5:  confidence {c1-c0:+.3f}, "
          f"accuracy {a1-a0:+.3f},  ratio {(c1-c0)/(a1-a0):.1f}x")
print()
for key, lab, *_ in SPEC:
    r = np.corrcoef(np.asarray(dec[key])[both], np.asarray(simple[key])[both])[0, 1]
    md = np.max(np.abs(np.asarray(dec[key])[both] - np.asarray(simple[key])[both]))
    print(f"  {lab:11s} r = {r:.3f}   max |difference| = {md:.3f}")
print("\nSaved figures/fig2_axis_comparison.pdf")
