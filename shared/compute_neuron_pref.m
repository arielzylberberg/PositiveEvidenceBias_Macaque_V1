%% compute_neuron_pref.m
%
% Classifies each neuron in each dataset as Right-preferring or
% Left-preferring using a two-sample t-test (right-choice vs left-choice
% trials) on z-scored spike counts, with FDR correction across neurons
% within each session.
%
% This is the SINGLE SOURCE OF TRUTH for neuron preference labels.
% Run this script once (or whenever parameters change) and all downstream
% analyses (run_neural_map.m, run_PSTH_by_choice_pref.m, …) will load
% the same classification.
%
% PARAMETERS
%   WIN_START  – start of counting window, seconds post-stimulus (default 0.050)
%   WIN_END    – end   of counting window, seconds post-stimulus (default 0.500)
%   ALPHA_SIG  – FDR-corrected significance level (default 0.05)
%
% OUTPUT FILE  (saved next to this script)
%   neuron_pref.mat
%     neuron_pref_data  {1 × ndatasets} cell array; each element is a struct:
%       .right_pref   [nneurons × 1 logical]  – sig. right-preferring neurons
%       .left_pref    [nneurons × 1 logical]  – sig. left-preferring neurons
%       .p_vals       [nneurons × 1 double]   – raw two-sample t-test p-values
%       .t_stats      [nneurons × 1 double]   – t-statistics (>0 → right-pref)
%       .sig_fdr      [nneurons × 1 logical]  – FDR-corrected significance mask
%     params  struct with fields WIN_START, WIN_END, ALPHA_SIG

clearvars -except masterNeuralStruct
close all

%% ===== PARAMETERS ====================================================
WIN_START  = 0.050;   % seconds post-stimulus onset
WIN_END    = 0.500;
ALPHA_SIG  = 0.05;    % FDR-corrected p-value threshold

%% ===== PATHS =========================================================
script_dir    = fileparts(mfilename('fullpath'));
repo_dir      = fullfile(script_dir, '..', '..');
structurePath = fullfile(repo_dir, 'Boundy-Singer_PNAS2024', 'Structures', filesep);
addpath(fullfile(script_dir, '..', 'functions'))

save_path = fullfile(script_dir, 'neuron_pref.mat');

if ~exist('masterNeuralStruct', 'var')
    fprintf('Loading masterNeuralStruct...\n');
    load(fullfile(structurePath, 'masterNeuralStruct_FULL.mat'));
end

ndatasets        = numel(masterNeuralStruct);
neuron_pref_data = cell(1, ndatasets);

fprintf('Computing neuron preferences (%d datasets)...\n', ndatasets);
fprintf('  Window: %.0f – %.0f ms  |  FDR alpha: %.2f\n\n', ...
        WIN_START*1000, WIN_END*1000, ALPHA_SIG);

%% ===== PER-SESSION LOOP ==============================================
for idataset = 1:ndatasets
    fprintf('Dataset %d / %d ... ', idataset, ndatasets);

    tmi = masterNeuralStruct{idataset}.behavior.trialMatrix.trialMatrix_index;
    tm  = masterNeuralStruct{idataset}.behavior.trialMatrix.trialMatrix;
    ds  = masterNeuralStruct{idataset}.dataStructTask;

    I         = ~isnan(ds.stimulus_onsets);
    spk_times = ds.unit_spike_times(I, :);
    nneurons  = size(spk_times, 2);

    choice_raw = tm(:, tmi.SIDE);        % −1=left, +1=right
    conf_raw   = tm(:, tmi.CONFIDENCE);  %  0=low,   1=high

    ntrials_beh = size(tm, 1);
    ntrials_neu = size(spk_times, 1);

    if ntrials_neu ~= ntrials_beh
        n           = min(ntrials_neu, ntrials_beh);
        spk_times   = spk_times(1:n, :);
        choice_raw  = choice_raw(1:n);
        conf_raw    = conf_raw(1:n);
        ntrials_neu = n;
    end

    valid = ismember(choice_raw, [-1,1]) & ismember(conf_raw, [0,1]);

    %% ---- Spike counts in [WIN_START, WIN_END] -----------------------
    counts = nan(ntrials_neu, nneurons);
    for iU = 1:nneurons
        for it = 1:ntrials_neu
            st = spk_times{it, iU};
            if isempty(st) || (isscalar(st) && isnan(st)), continue, end
            counts(it, iU) = sum(st >= WIN_START & st <= WIN_END);
        end
    end

    %% ---- Z-score per neuron ----------------------------------------
    counts_z = nan(size(counts));
    for iU = 1:nneurons
        col = counts(valid, iU);
        ok  = ~isnan(col);
        if sum(ok) < 10, continue, end
        mu = mean(col(ok));
        sg = std(col(ok));
        if sg == 0, continue, end
        counts_z(:, iU) = (counts(:, iU) - mu) / sg;
    end

    %% ---- Two-sample t-test: right-choice vs left-choice trials ------
    r_trials = valid & choice_raw ==  1;
    l_trials = valid & choice_raw == -1;

    p_vals  = nan(nneurons, 1);
    t_stats = nan(nneurons, 1);

    if sum(r_trials) >= 5 && sum(l_trials) >= 5
        for iU = 1:nneurons
            xr = counts_z(r_trials, iU);  xl = counts_z(l_trials, iU);
            xr = xr(~isnan(xr));          xl = xl(~isnan(xl));
            if numel(xr) < 5 || numel(xl) < 5, continue, end
            [~, p, ~, stats] = ttest2(xr, xl);
            p_vals(iU)  = p;
            t_stats(iU) = stats.tstat;
        end
    else
        fprintf('skipped (too few trials).\n');
        neuron_pref_data{idataset} = struct( ...
            'right_pref', false(nneurons,1), ...
            'left_pref',  false(nneurons,1), ...
            'p_vals',     p_vals,            ...
            't_stats',    t_stats,           ...
            'sig_fdr',    false(nneurons,1)  ...
        );
        continue
    end

    %% ---- FDR correction --------------------------------------------
    try
        sig_fdr = fdr_bh(p_vals', ALPHA_SIG);
    catch
        sig_fdr = p_vals' < ALPHA_SIG;
    end
    sig_fdr = sig_fdr(:);

    right_pref = sig_fdr & t_stats > 0;
    left_pref  = sig_fdr & t_stats < 0;

    fprintf('R-pref: %d  L-pref: %d\n', sum(right_pref), sum(left_pref));

    neuron_pref_data{idataset} = struct( ...
        'right_pref', right_pref, ...
        'left_pref',  left_pref,  ...
        'p_vals',     p_vals,     ...
        't_stats',    t_stats,    ...
        'sig_fdr',    sig_fdr     ...
    );
end

%% ===== SAVE ==========================================================
params = struct('WIN_START', WIN_START, 'WIN_END', WIN_END, 'ALPHA_SIG', ALPHA_SIG);

fprintf('\nSaving to: %s\n', save_path);
save(save_path, 'neuron_pref_data', 'params');
fprintf('Done. Load with:  load(''%s'');\n', save_path);
