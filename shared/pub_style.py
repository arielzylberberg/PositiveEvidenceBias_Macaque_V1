"""
Shared publication-figure style for every figure script in code_for_sharing/.

Mirrors style/pub_style.m (MATLAB) so that every figure in the paper --
whether rendered in MATLAB or Python -- uses the same font family, font
sizes, and colour palette. Edit THIS ONE FILE to change the look
everywhere it's used.

v2 (figure-consistency pass): every figure-producing script in
code_for_sharing/ now imports this module instead of re-declaring its own
local copy of font sizes and colours. Two concrete problems this fixed,
found by auditing every script rather than by eye:

  1. Five different font-size scales were in use across the figure set
     (e.g. Fig. 2 main panels used 10/12pt tick/label fonts, Fig. 2's own
     two supplement scripts used two OTHER, slightly different scales,
     and Fig. 1 panel D used a fourth) even though all of them claimed
     the same "Helvetica/Arial/DejaVu Sans" family. All now use the one
     scale defined below.
  2. Two colour pairs were not colour-vision-deficiency-safe, checked
     with a CVD-simulation validator rather than by eye (CIE Lab
     Delta-E between the two colours as seen by a deuteranope/protanope;
     >=8 is the target, <6 is a hard fail):
       - CHOSEN/UNCHOSEN (the chosen- vs unchosen-pool colour used in
         Fig. 4/5, i.e. the paper's central result) -- deutan Delta-E
         was 3.8. Replaced with a teal-green/burnt-orange pair,
         Delta-E 10.6.
       - ACCR_CORR/ACCR_INCR -- deutan Delta-E was 1.2. ACCR_INCR is not
         currently plotted alongside ACCR_CORR in any shipped figure
         (accuracy is shown as a single sequential ramp), so this was
         latent rather than an active bug, but is fixed for any future
         diverging use: ACCR_CORR is kept exactly as-is (it's "load
         bearing" as the sequential accuracy colormap in Fig. 2) and
         ACCR_INCR moved to a purple that separates cleanly from it
         (Delta-E 20.7).
     Also added CONTRAST_HI/CONTRAST_LO (used by Fig. 7/S1/S2, which
     colour-code BOTH a contrast split and a confidence split in the same
     multi-panel figure): the two scripts' pre-existing *_LO colours for
     "low contrast" and "low confidence" were both blues close enough
     together (Delta-E 5.8 -- indistinguishable even with normal colour
     vision) that the two dimensions could be confused in Fig. 7/S1/S2.
     CONTRAST_HI/LO is now a purple/green pair, chosen to stay clear of
     CONF_HIGH/CONF_LOW (which keep their original amber/steel-blue) even
     when both pairs appear in the same figure.

Used by (as of v2): every *.py script under fig1_task_and_behavior/,
fig2_gain_maps/, fig3_cross_decoder/, fig4_peb_pools/, fig5_simulation/,
and fig6_7_S1_S2_behavioral_models/ that calls matplotlib's savefig().
"""
import matplotlib as mpl

# ---- font sizes (pt); kept in sync with style/pub_style.m -----------------
# This is now the ONE scale used by every figure script (see module
# docstring: previously at least five slightly different scales were in
# use across the project).
FS_LBL = 13     # axis labels
FS_TICK = 11    # tick labels
FS_LEG = 11     # legend text
FS_TTL = 13     # panel titles (used sparingly -- most panels have none)
FS_PANEL = 17   # bold panel letters (A, B, C, ...)

# ---- colour palette (matches style/pub_style.m where a field exists) ------
# Confidence: warm amber-red (High) <-> cool steel-blue (Low)
CONF_HIGH = "#d64d12"
CONF_LOW = "#1461bd"

# Choice: teal-green (Right) <-> muted purple (Left)
CHOI_RIGHT = "#008768"
CHOI_LEFT = "#852e99"

# Accuracy: forest-green (correct) <-> purple (incorrect). ACCR_CORR is
# used alone today as a single-hue sequential ramp (Fig. 2); ACCR_INCR is
# for any future diverging correct-vs-incorrect use and is picked to
# separate cleanly from ACCR_CORR under deuteranopia/protanopia (v2 fix,
# see module docstring -- the previous brick-red ACCR_INCR failed that
# check, Delta-E 1.2).
ACCR_CORR = "#218c21"
ACCR_INCR = "#8b3a9e"

# Chosen/unchosen evidence pool (Fig. 4/5 -- the paper's central result).
# v2 fix (see module docstring): the previous green/salmon pair failed the
# deuteranopia check (Delta-E 3.8); this teal-green/burnt-orange pair
# passes (Delta-E 10.6) and keeps a similar "cool vs warm" feel.
CHOSEN = "#1f8f72"
UNCHOSEN = "#c1531b"

# Contrast condition, high vs low (Fig. 7/S1/S2). New in v2 (see module
# docstring): these panels also colour-code confidence (CONF_HIGH/LOW), so
# this pair is chosen to stay separable from CONF_HIGH/CONF_LOW as well as
# from each other. The CVD separation from CONF_LOW lands in the 6-8
# "warn" band (not a full pass) -- when both CONTRAST_* and CONF_* appear
# in the same panel, add a secondary encoding (e.g. solid vs dashed line)
# rather than relying on hue alone.
CONTRAST_HI = "#8a3fa0"
CONTRAST_LO = "#278a4f"

# Generic two-series comparison colour (e.g. AIC vs BIC bars in Fig. 6),
# for cases that aren't any of the semantic pairs above. New in v2 --
# previously Fig. 6 used matplotlib's unthemed default blue/red.
CMP_A = "#2b6ca3"
CMP_B = "#c1440e"

FONT_FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]


def apply_rcparams():
    """Apply the shared rcParams. Call once near the top of a figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "pdf.fonttype": 42,          # editable text in the PDF
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": FS_TICK,
        "axes.labelsize": FS_LBL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG,
    })


def despine(ax, keep=("left", "bottom")):
    """Remove all spines except those named in `keep` (default: top/right
    off, left/bottom on -- the look already used throughout the figure
    set). Call after plotting."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def panel_letter(ax, letter, x=-0.18, y=1.12):
    """Bold panel letter (A, B, C, ...) clear of the top-left corner.

    va="bottom" (rather than "top") anchors the text's baseline at y, so the
    letter sits entirely above y=1.0 (the axes top) with margin to spare --
    it cannot dip down and overlap the top spine/ticks/title of a panel that
    has one. clip_on=False guards against clipping when a panel's axes
    bounding box is tight against the figure edge.
    """
    ax.text(x, y, letter, transform=ax.transAxes, clip_on=False,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")
