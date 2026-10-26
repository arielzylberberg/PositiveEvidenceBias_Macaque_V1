% make_figure.m
%
% SUPERSEDED (v2 of this package): Fig. 1 panels B and C (the two panels
% this script's "FIGURE 1" block produces) are now generated in Python
% instead, by make_panels_bcd.py in this same folder, which also
% regenerates panel D and was verified to reproduce this script's own
% numbers exactly (same data, same no-intercept/intercept logistic fits,
% same quantile bin edges) while drawing all three panels from the one
% shared shared/pub_style.py palette. This removes the MATLAB/Octave
% dependency for Fig. 1 entirely. This file is kept only for provenance /
% in case MATLAB output is wanted for cross-checking -- it is not part of
% the recommended path to reproduce Fig. 1 in this package anymore. Its
% own "FIGURE 2" block (below) was already unused dead code before this
% change; see that block's own comment.
%
% FIGURE 1 – Three-panel figure:
%   Left   – p(correct)         vs |orientation|, split by contrast (median)
%   Middle – p(high confidence) vs |orientation|, split by contrast (median)
%   Right  – p(correct)         vs |orientation|, split by confidence (low/high)
%
% FIGURE 2 – Four-panel figure (2 × 2):
%   Top row    – p(high confidence) vs |orientation|, split by correct / incorrect,
%                separately for low contrast (left) and high contrast (right)
%   Bottom row – p(correct)         vs |orientation|, split by high / low confidence,
%                separately for low contrast (left) and high contrast (right)
%
% Data points: ~5 quantile bins of |orientation|, ± SEM
% Fits: logistic regression (glmfit) — naturally bounded [0,1] for both panels.
%       With |ori| as predictor the logistic is monotonically increasing,
%       which is exactly what we want for both correct & confidence.

clearvars; close all;

%% ── Paths ────────────────────────────────────────────────────────────────────
script_dir = fileparts(mfilename('fullpath'));
data_file  = fullfile(script_dir, 'data', 'data_behav.mat');
addpath(fullfile(script_dir, '..', 'shared'));
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% Shared publication style (fonts + colours), kept in sync with
% pub_style.py -- see shared/pub_style.m for the full v2 changelog.
S = pub_style();

%% ── Load data ────────────────────────────────────────────────────────────────
d        = load(data_file);
abs_ori  = abs(d.orientation);
contrast = d.contrast;
correct  = d.correct;       % 0 / 1
conf     = d.confidence;    % 0 / 1  (high = 1)

%% ── Median contrast split ────────────────────────────────────────────────────
med_c = median(contrast);
lo    = contrast <  med_c;   % low contrast
hi    = contrast >= med_c;   % high contrast
fprintf('Contrast median: %.4f  |  low n=%d  high n=%d\n', med_c, sum(lo), sum(hi));

%% ── Quantile bins of |orientation| (~5, across all trials) ──────────────────
N_BINS    = 5;
qedges    = quantile(abs_ori, linspace(0, 1, N_BINS + 1));
qedges(1) = 0;       % include exactly 0
qedges(end) = Inf;   % catch-all upper bin

%% ── Logistic fits (binomial GLM = logistic regression) ──────────────────────
[ap_lo, bp_lo] = logfit_noint(abs_ori(lo), correct(lo));
[ap_hi, bp_hi] = logfit_noint(abs_ori(hi), correct(hi));

[ac_lo, bc_lo] = logfit(abs_ori(lo), conf(lo));
[ac_hi, bc_hi] = logfit(abs_ori(hi), conf(hi));

fprintf('Psychometric  – low:  a=%.3f  b=%.4f\n', ap_lo, bp_lo);
fprintf('Psychometric  – high: a=%.3f  b=%.4f\n', ap_hi, bp_hi);
fprintf('Confidence    – low:  a=%.3f  b=%.4f\n', ac_lo, bc_lo);
fprintf('Confidence    – high: a=%.3f  b=%.4f\n', ac_hi, bc_hi);

%% ── Confidence split (pooled across contrast) ────────────────────────────────
hconf = logical(conf);
lconf = ~logical(conf);
fprintf('Confidence split – high conf n=%d  low conf n=%d\n', sum(hconf), sum(lconf));

% p(correct), split by high/low confidence — no intercept: p(correct)=0.5 at |ori|=0
[ap_h, bp_h] = logfit_noint(abs_ori(hconf), correct(hconf));
[ap_l, bp_l] = logfit_noint(abs_ori(lconf), correct(lconf));
fprintf('Psychometric (conf split) – high conf: a=%.3f  b=%.4f\n', ap_h, bp_h);
fprintf('Psychometric (conf split) – low conf:  a=%.3f  b=%.4f\n', ap_l, bp_l);

%% ── Smooth x-axis for fit curves ─────────────────────────────────────────────
x_fit = linspace(0, max(abs_ori), 300)';
sigm  = @(a, b, x) 1 ./ (1 + exp(-(a + b .* x)));

%% ── Colors ───────────────────────────────────────────────────────────────────
% v2: previously drawn from movshon_colors(3) (a 5-colour categorical
% palette, Goris Neuron 2015) for contrast and from two more one-off
% hardcoded pairs for confidence; all three now come from the shared
% pub_style() palette so this figure's contrast/confidence colours match
% Fig. 7/S1/S2 (which colour-code the same two dimensions) instead of
% each figure inventing its own. See shared/pub_style.m for the Delta-E
% rationale. (movshon_colors.m is left in shared/ unused, in case other,
% unregenerated lab figures still reference it.)
COL_LO = S.CONTRAST_LO;
COL_HI = S.CONTRAST_HI;

COL_HCONF = S.CONF_HIGH;   % high confidence
COL_LCONF = S.CONF_LOW;    % low confidence

%% ── Figure ───────────────────────────────────────────────────────────────────
fig = figure('Units', 'centimeters', 'Position', [2 2 29 8], 'Color', 'w');
tl  = tiledlayout(1, 3, 'TileSpacing', 'loose', 'Padding', 'compact');

% ── Panel 1: Psychometric ────────────────────────────────────────────────────
nexttile; hold on;

draw_panel(abs_ori, correct, qedges, ...
    lo, hi, ...
    ap_lo, bp_lo, ap_hi, bp_hi, ...
    x_fit, sigm, COL_LO, COL_HI);

yline(0.5, '--', 'Color', [.7 .7 .7], 'LineWidth', .8);
ylabel('p(correct)', 'FontSize', S.FS_LBL);
title('Psychometric function', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);
ylim([0.4 1.04]);

% Legend on first panel only
h_lo = plot(nan, nan, 'o-', 'Color', COL_LO, 'MarkerFaceColor', COL_LO, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
h_hi = plot(nan, nan, 'o-', 'Color', COL_HI, 'MarkerFaceColor', COL_HI, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
legend([h_lo h_hi], {'Low contrast', 'High contrast'}, ...
    'Location', 'southeast', 'Box', 'off', 'FontSize', S.FS_LEG);

% ── Panel 2: Confidence ──────────────────────────────────────────────────────
nexttile; hold on;

draw_panel(abs_ori, conf, qedges, ...
    lo, hi, ...
    ac_lo, bc_lo, ac_hi, bc_hi, ...
    x_fit, sigm, COL_LO, COL_HI);

ylabel('p(high confidence)', 'FontSize', S.FS_LBL);
title('Confidence', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);
ylim([0 1.04]);

% ── Panel 3: Psychometric split by confidence ───────────────────────────────
nexttile; hold on;

draw_panel2(abs_ori, correct, qedges, ...
    hconf, COL_HCONF, [ap_h, bp_h], ...
    lconf, COL_LCONF, [ap_l, bp_l], ...
    x_fit, sigm);

yline(0.5, '--', 'Color', [.7 .7 .7], 'LineWidth', .8);
ylabel('p(correct)', 'FontSize', S.FS_LBL);
title('Confidence split', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);
ylim([0.4 1.04]);

h_hc = plot(nan, nan, 'o-', 'Color', COL_HCONF, 'MarkerFaceColor', COL_HCONF, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
h_lc = plot(nan, nan, 'o-', 'Color', COL_LCONF, 'MarkerFaceColor', COL_LCONF, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
legend([h_hc h_lc], {'High confidence', 'Low confidence'}, ...
    'Location', 'southeast', 'Box', 'off', 'FontSize', S.FS_LEG);

% ── Shared axes formatting ────────────────────────────────────────────────────
for ax = findobj(fig, 'Type', 'axes')'
    ax.Box     = 'off';
    ax.TickDir = 'out';
    ax.XLim    = [0 20];
    ax.FontSize = S.FS_TICK;
    xlabel(ax, '|Orientation| (°)', 'FontSize', S.FS_LBL);
end

%% ── Save figure 1 ────────────────────────────────────────────────────────────
exportgraphics(fig, fullfile(fig_dir, 'psychometric_confidence.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(fig_dir, 'psychometric_confidence.png'), 'Resolution', 150);
fprintf('Saved figure 1 to %s\n', fig_dir);


%% ════════════════════════════════════════════════════════════════════════════
%% ── FIGURE 2 : confidence × accuracy split ───────────────────────────────
%% ════════════════════════════════════════════════════════════════════════════
%
%   Top row    – p(high confidence) vs |ori|, correct vs incorrect trials,
%                split by contrast (low | high)
%   Bottom row – p(correct)         vs |ori|, high vs low confidence trials,
%                split by contrast (low | high)

%% ── Colors for figure 2 ──────────────────────────────────────────────────────
% v2: now pub_style()'s ACCR_CORR/ACCR_INCR instead of a one-off
% green/orange pair, so "correct vs incorrect" uses the same colours
% everywhere it's drawn as a diverging pair.
COL_CORR  = S.ACCR_CORR;     % correct
COL_INCORR= S.ACCR_INCR;     % incorrect
% COL_HCONF / COL_LCONF defined above (used for Figure 1, panel 3, and reused here)

%% ── Subgroup masks ───────────────────────────────────────────────────────────
lo_corr   = lo & logical(correct);
lo_incorr = lo & ~logical(correct);
hi_corr   = hi & logical(correct);
hi_incorr = hi & ~logical(correct);

lo_hconf  = lo & logical(conf);
lo_lconf  = lo & ~logical(conf);
hi_hconf  = hi & logical(conf);
hi_lconf  = hi & ~logical(conf);

fprintf('Low contrast  – correct: %d  incorrect: %d\n',  sum(lo_corr),  sum(lo_incorr));
fprintf('High contrast – correct: %d  incorrect: %d\n',  sum(hi_corr),  sum(hi_incorr));
fprintf('Low contrast  – high conf: %d  low conf: %d\n', sum(lo_hconf), sum(lo_lconf));
fprintf('High contrast – high conf: %d  low conf: %d\n', sum(hi_hconf), sum(hi_lconf));

%% ── Logistic fits for figure 2 ───────────────────────────────────────────────
% p(high confidence), split by correct/incorrect within each contrast
[ac_loc, bc_loc]   = logfit(abs_ori(lo_corr),   conf(lo_corr));
[ac_loi, bc_loi]   = logfit(abs_ori(lo_incorr), conf(lo_incorr));
[ac_hic, bc_hic]   = logfit(abs_ori(hi_corr),   conf(hi_corr));
[ac_hii, bc_hii]   = logfit(abs_ori(hi_incorr), conf(hi_incorr));

% p(correct), split by high/low confidence within each contrast
% No intercept: forces p(correct)=0.5 at |ori|=0
[ap_loh, bp_loh]   = logfit_noint(abs_ori(lo_hconf), correct(lo_hconf));
[ap_lol, bp_lol]   = logfit_noint(abs_ori(lo_lconf), correct(lo_lconf));
[ap_hih, bp_hih]   = logfit_noint(abs_ori(hi_hconf), correct(hi_hconf));
[ap_hil, bp_hil]   = logfit_noint(abs_ori(hi_lconf), correct(hi_lconf));

%% ── Figure 2 ─────────────────────────────────────────────────────────────────
fig2 = figure('Units', 'centimeters', 'Position', [2 2 24 16], 'Color', 'w');
tl2  = tiledlayout(2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');
title(tl2, 'Confidence × accuracy split', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);

% ── Panel (1,1): p(high conf) – low contrast ─────────────────────────────────
nexttile(1); hold on;

draw_panel2(abs_ori, conf, qedges, ...
    lo_corr,   COL_CORR,   [ac_loc, bc_loc], ...
    lo_incorr, COL_INCORR, [ac_loi, bc_loi], ...
    x_fit, sigm);

ylabel('p(high confidence)', 'FontSize', S.FS_LBL);
title('Low contrast', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);
ylim([0 1.04]);

h_c  = plot(nan, nan, 'o-', 'Color', COL_CORR,   'MarkerFaceColor', COL_CORR,   'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
h_i  = plot(nan, nan, 'o-', 'Color', COL_INCORR, 'MarkerFaceColor', COL_INCORR, 'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
legend([h_c h_i], {'Correct', 'Incorrect'}, 'Location', 'northwest', 'Box', 'off', 'FontSize', S.FS_LEG);

% ── Panel (1,2): p(high conf) – high contrast ─────────────────────────────────
nexttile(2); hold on;

draw_panel2(abs_ori, conf, qedges, ...
    hi_corr,   COL_CORR,   [ac_hic, bc_hic], ...
    hi_incorr, COL_INCORR, [ac_hii, bc_hii], ...
    x_fit, sigm);

title('High contrast', 'FontWeight', 'bold', 'FontSize', S.FS_TTL);
ylim([0 1.04]);

% ── Panel (2,1): p(correct) – low contrast ────────────────────────────────────
nexttile(3); hold on;

draw_panel2(abs_ori, correct, qedges, ...
    lo_hconf, COL_HCONF, [ap_loh, bp_loh], ...
    lo_lconf, COL_LCONF, [ap_lol, bp_lol], ...
    x_fit, sigm);

yline(0.5, '--', 'Color', [.7 .7 .7], 'LineWidth', .8);
ylabel('p(correct)', 'FontSize', S.FS_LBL);
ylim([0.3 1.04]);

h_hc = plot(nan, nan, 'o-', 'Color', COL_HCONF, 'MarkerFaceColor', COL_HCONF, 'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
h_lc = plot(nan, nan, 'o-', 'Color', COL_LCONF, 'MarkerFaceColor', COL_LCONF, 'MarkerEdgeColor', 'w', 'LineWidth', 2, 'MarkerSize', 7);
legend([h_hc h_lc], {'High confidence', 'Low confidence'}, 'Location', 'southeast', 'Box', 'off', 'FontSize', S.FS_LEG);

% ── Panel (2,2): p(correct) – high contrast ───────────────────────────────────
nexttile(4); hold on;

draw_panel2(abs_ori, correct, qedges, ...
    hi_hconf, COL_HCONF, [ap_hih, bp_hih], ...
    hi_lconf, COL_LCONF, [ap_hil, bp_hil], ...
    x_fit, sigm);

yline(0.5, '--', 'Color', [.7 .7 .7], 'LineWidth', .8);
ylim([0.3 1.04]);

% ── Shared axes formatting ────────────────────────────────────────────────────
for ax = findobj(fig2, 'Type', 'axes')'
    ax.Box     = 'off';
    ax.TickDir = 'out';
    ax.XLim    = [0 20];
    ax.FontSize = S.FS_TICK;
    xlabel(ax, '|Orientation| (°)', 'FontSize', S.FS_LBL);
end

%% ── Save figure 2 ────────────────────────────────────────────────────────────
exportgraphics(fig2, fullfile(fig_dir, 'conf_accuracy_split.pdf'), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(fig_dir, 'conf_accuracy_split.png'), 'Resolution', 150);
fprintf('Saved figure 2 to %s\n', fig_dir);


%% ════════════════════════ Local functions ════════════════════════════════════

function [a, b] = logfit(x, y)
    % Logistic regression: p = sigmoid(a + b*x) via binomial GLM
    beta = glmfit(x, y, 'binomial', 'link', 'logit');
    a = beta(1);
    b = beta(2);
end

function [a, b] = logfit_noint(x, y)
    % Logistic regression with no intercept: forces p=0.5 at x=0
    % p = sigmoid(b*x),  i.e. a is fixed at 0
    beta = glmfit(x, y, 'binomial', 'link', 'logit', 'constant', 'off');
    a = 0;
    b = beta(1);
end

function draw_panel(abs_ori, y, edges, lo, hi, a_lo, b_lo, a_hi, b_hi, x_fit, sigm, COL_LO, COL_HI)
    groups = {lo,    hi   };
    cols   = {COL_LO, COL_HI};
    params = {[a_lo, b_lo], [a_hi, b_hi]};

    for g = 1:2
        mask = groups{g};
        col  = cols{g};
        a    = params{g}(1);
        b    = params{g}(2);

        % Smooth fit line
        plot(x_fit, sigm(a, b, x_fit), '-', 'Color', col, 'LineWidth', 2);

        % Binned data ± SEM
        [cx, mn, se] = bin_data(abs_ori(mask), y(mask), edges);
        errorbar(cx, mn, se, 'o', ...
            'Color',           col, ...
            'MarkerFaceColor', col, ...
            'MarkerEdgeColor', 'w', ...
            'MarkerSize',      7,   ...
            'LineWidth',       1.2, ...
            'CapSize',         4);
    end
end

function [cx, mn, se] = bin_data(x, y, edges)
    n  = numel(edges) - 1;
    cx = NaN(1, n);
    mn = NaN(1, n);
    se = NaN(1, n);
    for i = 1:n
        if i < n
            m = x >= edges(i) & x < edges(i+1);
        else
            m = x >= edges(i);   % include upper edge in last bin
        end
        if sum(m) < 3, continue; end
        vals  = y(m);
        cx(i) = mean(x(m));
        mn(i) = mean(vals);
        se(i) = std(vals, 0) / sqrt(sum(m));
    end
    ok = ~isnan(cx);
    cx = cx(ok);
    mn = mn(ok);
    se = se(ok);
end

function draw_panel2(abs_ori, y, edges, mask1, col1, params1, mask2, col2, params2, x_fit, sigm)
% Draw two groups (e.g. correct vs incorrect, or high vs low conf) on the
% current axes using logistic fit lines + binned data points.
%   mask1/2  – logical index for each group
%   col1/2   – RGB colour
%   params1/2 – [a, b] logistic parameters

    groups = {mask1,   mask2  };
    cols   = {col1,    col2   };
    params = {params1, params2};

    for g = 1:2
        mask = groups{g};
        col  = cols{g};
        a    = params{g}(1);
        b    = params{g}(2);

        % Smooth fit line
        plot(x_fit, sigm(a, b, x_fit), '-', 'Color', col, 'LineWidth', 2);

        % Binned data ± SEM
        [cx, mn, se] = bin_data(abs_ori(mask), y(mask), edges);
        errorbar(cx, mn, se, 'o', ...
            'Color',           col, ...
            'MarkerFaceColor', col, ...
            'MarkerEdgeColor', 'w', ...
            'MarkerSize',      7,   ...
            'LineWidth',       1.2, ...
            'CapSize',         4);
    end
end
