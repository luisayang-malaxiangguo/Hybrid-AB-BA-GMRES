% save_all_figures.m
% ----------------------------------------
% Finds all open figures and exports each
% as a vector PDF into the "fig" subfolder.
%
% Usage: 
%   1) Generate or open your figures in MATLAB.
%   2) Run this script: >> save_all_figures
%   3) Upload the created "fig" folder to Overleaf.
% ----------------------------------------

function save_all_figures()
    %--- 1. Set up output directory ---
    outputDir = fullfile(pwd, 'fig');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    %--- 2. Gather all open figure handles ---
    figHandles = findall(0, 'Type', 'figure');

    %--- 3. Loop and export each as PDF ---
    for k = 1:numel(figHandles)
        fig = figHandles(k);

        % Ensure paper size matches figure size
        set(fig, 'Units', 'Inches');
        figPos = get(fig, 'Position');
        set(fig, 'PaperUnits', 'Inches', ...
                 'PaperPosition', [0 0 figPos(3) figPos(4)], ...
                 'PaperSize',   [figPos(3) figPos(4)]);

        % Build filename: Figure<number>.pdf
        filename = fullfile(outputDir, sprintf('Figure%d.pdf', fig.Number));

        % Export vector PDF
        exportgraphics(fig, filename, 'ContentType', 'vector');

        fprintf('Saved Figure %d → %s\n', fig.Number, filename);
    end

    fprintf('All figures exported to folder:\n  %s\n', outputDir);
end
