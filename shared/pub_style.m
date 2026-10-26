function S = pub_style()
%PUB_STYLE  Shared publication-figure style for code_for_sharing2/.
%
%   S = pub_style()
%
%   MATLAB mirror of shared/pub_style.py -- the two files are kept in sync
%   field-for-field (same hex values, just expressed as 0-1 RGB triples
%   here). Edit THIS ONE FILE (and its Python twin) to change the look
%   everywhere it's used, instead of duplicating literal colour/font
%   values inside each plotting script.
%
%   v2 (figure-consistency pass, see pub_style.py's module docstring for
%   the full rationale/Delta-E numbers): this version is new relative to
%   the original analysis_az/style/pub_style.m --
%     - ACCR_INCR changed from brick-red [0.80 0.10 0.15] to purple
%       [0.55 0.23 0.62] (deutan/protan Delta-E vs ACCR_CORR was 1.2,
%       now 20.7). ACCR_CORR is unchanged (it's the sequential accuracy
%       colormap in Fig. 2 and is "load bearing").
%     - Added CHOSEN/UNCHOSEN (Fig. 4/5 -- the paper's central result;
%       the previous ad hoc green/salmon pair used directly in those
%       scripts failed the deuteranopia check, Delta-E 3.8; this
%       teal-green/burnt-orange pair passes, Delta-E 10.6).
%     - Added CONTRAST_HI/CONTRAST_LO (Fig. 1 panels B/C and Fig.
%       7/S1/S2, which also colour-code confidence in the same
%       multi-panel figure -- the previous per-script contrast colours,
%       including this file's own movshon_colors(3) pair, were each
%       invented independently and one of them collided with
%       CONF_LOW, Delta-E 5.8).
%     - Added CMP_A/CMP_B, a generic two-series comparison colour (Fig.
%       6 AIC-vs-BIC bars, previously MATLAB/matplotlib default blue/red).
%     - Added FS_LEG (legend text size), to match pub_style.py.
%
%   Fields:
%     CONF_HIGH / CONF_LOW       - confidence palette (warm amber-red / cool steel-blue)
%     CHOI_RIGHT / CHOI_LEFT     - choice palette (teal-green / muted purple)
%     ACCR_CORR / ACCR_INCR      - accuracy palette (forest-green / purple)
%     CHOSEN / UNCHOSEN          - chosen- vs unchosen-pool palette (Fig. 4/5)
%     CONTRAST_HI / CONTRAST_LO  - contrast-condition palette (Fig. 1B/C, 7, S1/S2)
%     CMP_A / CMP_B              - generic two-series comparison palette (Fig. 6)
%     FS_LBL / FS_TICK / FS_LEG / FS_TTL / FS_PANEL - font sizes (axis
%                                  labels, tick labels, legend text,
%                                  titles, panel letters A/B/C...)
%     LW_CONTOUR / LW_AXIS       - line widths for contour lines / axis box
%     FONT_NAME                  - font family for all text
%
%   Used by (as of v2): fig1_task_and_behavior/run_make_figure.m. (Every
%   other figure is produced by the Python scripts, which use
%   pub_style.py instead.)

% Confidence: warm amber-red (High) <-> cool steel-blue (Low)
S.CONF_HIGH  = [0.84 0.30 0.07];
S.CONF_LOW   = [0.08 0.38 0.74];

% Choice: teal-green (Right) <-> muted purple (Left)
S.CHOI_RIGHT = [0.00 0.53 0.42];
S.CHOI_LEFT  = [0.52 0.18 0.60];

% Accuracy: forest-green (correct) <-> purple (incorrect; used only if a
% diverging accuracy scale is needed -- the pub accuracy map itself uses
% a sequential white->green scale, see add_seq_colorbar). ACCR_INCR
% changed in v2, see module docstring.
S.ACCR_CORR  = [0.13 0.55 0.13];
S.ACCR_INCR  = [0.55 0.23 0.62];

% Chosen/unchosen evidence pool (Fig. 4/5 -- the paper's central result).
% New in v2, see module docstring.
S.CHOSEN     = [0.12 0.56 0.45];
S.UNCHOSEN   = [0.76 0.33 0.11];

% Contrast condition, high vs low (Fig. 1B/C, Fig. 7/S1/S2). New in v2,
% see module docstring -- replaces this file's own former movshon_colors
% categorical pair for contrast.
S.CONTRAST_HI = [0.54 0.25 0.63];
S.CONTRAST_LO = [0.15 0.54 0.31];

% Generic two-series comparison colour (e.g. AIC vs BIC bars in Fig. 6).
% New in v2.
S.CMP_A = [0.17 0.42 0.64];
S.CMP_B = [0.76 0.27 0.05];

% Font sizes (pt)
S.FS_LBL   = 13;   % axis labels
S.FS_TICK  = 11;   % tick labels
S.FS_LEG   = 11;   % legend text (new in v2, matches pub_style.py)
S.FS_TTL   = 13;   % panel titles / sgtitle
S.FS_PANEL = 17;   % panel letter labels (A, B, C, ...)

% Line widths
S.LW_CONTOUR = 1.4;
S.LW_AXIS    = 0.8;

% Font family
S.FONT_NAME = 'Helvetica';
end
