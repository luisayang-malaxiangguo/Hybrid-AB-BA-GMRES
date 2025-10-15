function plot_error_vs_mismatch_norm()
%This generates a log-log plot showing the final relative
% solution error as a function of the mismatch norm ||B - A'||


clear all; clc; close all;

%% 1) Set up Test Problem, Noise, and Base Perturbation
fprintf('1. Setting up the test problem...\n');
n = 32;
problem_name = 'heat';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

rng(0); % For reproducibility
noise_level = 1e-2;
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b_noise = b_exact + noise;

%  Create a base random perturbation matrix E 
% We will scale this matrix to control the mismatch norm.
E = randn(size(A'));
E = E / norm(E, 'fro'); % Normalize E to have a Frobenius norm of 1.

%  Solver Parameters 
maxit = n;
tol = 1e-6;

%% 2) Define Perturbation Levels and Prepare for Loop
fprintf('2. Preparing for perturbation analysis loop...\n');

%  Define a range of scaling factors for the perturbation 
c_range = logspace(-8, -1, 20);

%  Initialize storage for results 
mismatch_norms = zeros(size(c_range));
final_errors_ab = zeros(size(c_range));
final_errors_ba = zeros(size(c_range));

%% 3) Loop Through Perturbation Levels
fprintf('3. Running analysis for %d perturbation levels...\n', length(c_range));
tic; % Start a timer

for i = 1:length(c_range)
    c = c_range(i);
    
    %  Create the perturbation for this level 
    current_perturbation = c * E;
    B_pert = A' + current_perturbation;
    
    %  Store the mismatch norm (the x-axis value) 
    mismatch_norms(i) = norm(current_perturbation, 'fro');
    
    % Define the DeltaM terms for the solvers
    DeltaM_AB = A * current_perturbation;
    DeltaM_BA = current_perturbation * A;
    
    %  Find optimal lambda for each method at this perturbation level 
    k_gcv = 20;
    m = size(A,1);
    options = optimset('Display', 'off', 'TolX', 1e-8);

    gcv_handle_ab = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ab');
    lambda_gcv_ab = fminbnd(gcv_handle_ab, 1e-9, 1e-1, options);
    
    gcv_handle_ba = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ba');
    lambda_gcv_ba = fminbnd(gcv_handle_ba, 1e-9, 1e-1, options);
    
    %  Solve with each hybrid method and store the final error 
    [~, err_hist_ab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ab, DeltaM_AB);
    final_errors_ab(i) = err_hist_ab(end);
    
    [~, err_hist_ba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ba, DeltaM_BA);
    final_errors_ba(i) = err_hist_ba(end);
    
    fprintf('   - Level %d/%d complete. Mismatch Norm: %.2e, Error (AB): %.3f, Error (BA): %.3f\n', ...
            i, length(c_range), mismatch_norms(i), final_errors_ab(i), final_errors_ba(i));
end
toc; % Stop the timer

%% 4) Generate the Final Plot
fprintf('4. Generating the final plot...\n');

figure('Name', 'Error vs. Mismatch Norm', 'Position', [300 300 800 600]);
loglog(mismatch_norms, final_errors_ab, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid AB-GMRES');
hold on;
loglog(mismatch_norms, final_errors_ba, 'r-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid BA-GMRES');
hold off;

grid on;
title('Final Solution Error vs. Back-Projector Mismatch Norm');
xlabel('Mismatch Norm ||B - A^T||_F');
ylabel('Final Relative Error ||x_k - x_{true}|| / ||x_{true}||');
legend('show', 'Location', 'NorthWest');
axis tight;
set(gca, 'FontSize', 12);

fprintf(' Analysis complete. \n');
end
