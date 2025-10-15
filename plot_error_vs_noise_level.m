function plot_error_vs_noise_level()

% This script generates a log-log plot showing the final relative
% solution error as a function of the noise level.

clear all; clc; close all

%% 1) Set up Test Problem and Parameters
fprintf('1. Setting up the test problem...\n');
n = 32;
problem_name = 'shaw';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

%  Use a fixed, small perturbation for the back-projector 
rng(0); % For reproducibility
E = 1e-4 * randn(size(A'));
B_pert = A' + E;
DeltaM_AB = A * E;
DeltaM_BA = E * A;

%  Solver Parameters 
maxit = n;
tol = 1e-6;

%% 2) Define Noise Levels and Prepare for Loop
fprintf('2. Preparing for noise level analysis loop...\n');

%  Define a range of relative noise levels (e.g., 0.01% to 10%) 
noise_levels = logspace(-4, -1, 20);

%  Initialize storage for results 
final_errors_hab = zeros(size(noise_levels)); % Hybrid AB
final_errors_hba = zeros(size(noise_levels)); % Hybrid BA
final_errors_nab = zeros(size(noise_levels)); % Non-hybrid AB
final_errors_nba = zeros(size(noise_levels)); % Non-hybrid BA

%% 3) Loop Through Each Noise Level
fprintf('3. Running analysis for %d noise levels...\n', length(noise_levels));
tic;

for i = 1:length(noise_levels)
    level = noise_levels(i);
    
    %  Create the noisy data vector 'b' for this level 
    noise = randn(size(b_exact));
    b_noise = b_exact + (noise / norm(noise)) * level * norm(b_exact);
    
    %  Run Hybrid Methods (with GCV to find optimal lambda for this noise level) 
    k_gcv = 20;
    m = size(A,1);
    options = optimset('Display', 'off', 'TolX', 1e-8);

    gcv_handle_ab = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ab');
    lambda_gcv_ab = fminbnd(gcv_handle_ab, 1e-9, 1e-1, options);
    [~, err_hab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ab, DeltaM_AB);
    final_errors_hab(i) = err_hab(end);
    
    gcv_handle_ba = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ba');
    lambda_gcv_ba = fminbnd(gcv_handle_ba, 1e-9, 1e-1, options);
    [~, err_hba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ba, DeltaM_BA);
    final_errors_hba(i) = err_hba(end);
    
    %  Run Non-Hybrid Methods 
    [~, err_nab, ~, ~] = ABgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_AB);
    final_errors_nab(i) = err_nab(end);
    
    [~, err_nba, ~, ~] = BAgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_BA);
    final_errors_nba(i) = err_nba(end);
    
    fprintf('   - Level %d/%d complete. Noise Level: %.2e%%\n', i, length(noise_levels), level*100);
end
toc;

%% 4) Generate the Final Plot
fprintf('4. Generating the final plot...\n');

figure('Name', 'Error vs. Noise Level (All Methods)', 'Position', [300 300 800 600]);
% Plot Hybrid Methods (Solid Lines)
loglog(noise_levels, final_errors_hab, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid AB');
hold on;
loglog(noise_levels, final_errors_hba, 'r-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid BA');

% Plot Non-Hybrid Methods (Dashed Lines)
loglog(noise_levels, final_errors_nab, 'b--s', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Non-Hybrid AB');
loglog(noise_levels, final_errors_nba, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Non-Hybrid BA');
hold off;

grid on;
title('Final Solution Error vs. Data Noise Level');
xlabel('Relative Noise Level ||e|| / ||b_{exact}||');
ylabel('Final Relative Error ||x_k - x_{true}|| / ||x_{true}||');
legend('show', 'Location', 'NorthWest');
axis tight;
set(gca, 'FontSize', 12);

fprintf(' Analysis complete. \n');
end