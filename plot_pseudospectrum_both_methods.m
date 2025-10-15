function plot_pseudospectrum_both_methods()
% PLOT_PSEUDOSPECTRUM_BOTH_METHODS Computes and plots the pseudospectrum
% for both non-symmetric system matrices M = B*A and M = A*B.
% It produces a separate figure for each matrix for clarity.

%% 0) Clean Slate
clear all;
clc;

%% 1) Set up the Non-Symmetric Matrices
fprintf('1. Setting up the non-symmetric system matrices...\n');
n = 32;
problem_name = 'shaw';
[A, ~, ~] = generate_test_problem(problem_name, n);

% --- Create a significant perturbation to show non-normality ---
rng(0); % For reproducibility
perturbation_level = 0.1; % Use a larger perturbation for a more interesting plot
E = randn(size(A'));
E = E / norm(E, 'fro') * perturbation_level;
B_pert = A' + E;

% The two matrices of interest for the pseudospectra
M_ba = B_pert * A;
M_ab = A * B_pert;

%% 2) Compute and Plot Pseudospectrum for M = BA
fprintf('2. Computing and plotting pseudospectrum for M = BA...\n');
generate_pseudospectrum_plot(M_ba, 'Pseudospectrum of M = BA');

%% 3) Compute and Plot Pseudospectrum for M = AB
fprintf('3. Computing and plotting pseudospectrum for M = AB...\n');
generate_pseudospectrum_plot(M_ab, 'Pseudospectrum of M = AB');

fprintf('--- Analysis complete. ---\n');
end


% --- HELPER FUNCTION to generate a single pseudospectrum plot ---
function generate_pseudospectrum_plot(M, figure_title)
    % --- Define Grid for Pseudospectrum Calculation ---
    eigenvalues = eig(M);
    
    % Define the boundaries of the grid around the eigenvalues
    real_min = min(real(eigenvalues)) - 0.1;
    real_max = max(real(eigenvalues)) + 0.1;
    imag_min = min(imag(eigenvalues)) - 0.1;
    imag_max = max(imag(eigenvalues)) + 0.1;

    % Create the grid
    grid_points = 150; % Higher number gives a smoother plot but is slower
    x = linspace(real_min, real_max, grid_points);
    y = linspace(imag_min, imag_max, grid_points);
    [X, Y] = meshgrid(x, y);
    Z = X + 1i * Y; % The grid in the complex plane

    % --- Compute the Pseudospectrum ---
    fprintf('   - Computing resolvent norm on a %d x %d grid...\n', grid_points, grid_points);
    tic;
    s_min = zeros(size(Z));
    I = eye(size(M));
    for i = 1:numel(Z)
        s_min(i) = svds(Z(i) * I - M, 1, 'smallest');
    end
    toc;

    % --- Generate the Plot ---
    figure('Name', figure_title);
    
    levels = [1e-1, 1e-2, 1e-3, 1e-4];
    contour(x, y, s_min, levels, 'LineWidth', 1.5);
    hold on;
    plot(real(eigenvalues), imag(eigenvalues), 'k.', 'MarkerSize', 15, 'DisplayName', 'Eigenvalues');
    hold off;

    grid on;
    colorbar;
    c = colorbar;
    ylabel(c, 'Smallest Singular Value \sigma_{min}(zI - M)');
    title(figure_title);
    xlabel('Real Part');
    ylabel('Imaginary Part');
    axis equal;
    legend('show', 'Location', 'NorthWest');
    set(gca, 'FontSize', 12);
end