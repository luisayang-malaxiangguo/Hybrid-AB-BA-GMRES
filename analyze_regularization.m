function analyze_regularization()
% This script produces four main figures:
%   1. L-Curve and Error-vs-Lambda analysis for Hybrid AB-GMRES.
%   2. L-Curve and Error-vs-Lambda analysis for Hybrid BA-GMRES.
%   3. A visual comparison of the solutions from both HYBRID methods.
%   4. A final visual comparison of solutions from ALL FOUR methods.
%
%% 1) Set up Test Problem & Parameters
fprintf('1. Setting up the test problem...\n');
n = 32;
problem_name = 'shaw';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);
% --- Add Noise and Perturbation ---
rng(0); % For reproducibility
noise_level = 1e-2; % A bit of noise makes regularization more important
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b = b_exact + noise; % Use the noisy right-hand side
E = 1e-4 * randn(size(A')); % A non-trivial perturbation
B_pert = A' + E;
DeltaM_AB = A * E; % Perturbation term for AB-GMRES
DeltaM_BA = E * A; % Perturbation term for BA-GMRES
% --- Solver Parameters ---
maxit = 32; % Run to full dimension for analysis
tol = 1e-6;
lambda_range = logspace(-10, 0, 100);
%% 2) Loop Through Lambda Range to Collect Data for Plots
fprintf('2. Running solvers for a range of lambda values to generate plot data...\n');
% --- Initialize storage for results ---
res_norms_ab = zeros(size(lambda_range)); sol_norms_ab = zeros(size(lambda_range)); err_norms_ab = zeros(size(lambda_range));
res_norms_ba = zeros(size(lambda_range)); sol_norms_ba = zeros(size(lambda_range)); err_norms_ba = zeros(size(lambda_range));
for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    % Solve with Hybrid AB-GMRES
    [x_ab, err_ab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b, x_true, tol, maxit, lambda, DeltaM_AB);
    res_norms_ab(i) = norm(b - A * x_ab) / norm(b);
    sol_norms_ab(i) = norm(x_ab);
    err_norms_ab(i) = err_ab(end);
    
    % Solve with Hybrid BA-GMRES
    [x_ba, err_ba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b, x_true, tol, maxit, lambda, DeltaM_BA);
    res_norms_ba(i) = norm(b - A * x_ba) / norm(b);
    sol_norms_ba(i) = norm(x_ba);
    err_norms_ba(i) = err_ba(end);
end
%% 3) Find Optimal Lambdas for Both Hybrid Methods
fprintf('3. Finding optimal lambdas for each hybrid method...\n');
k_gcv = 20; % Iteration for GCV function evaluation
m = size(A,1);
options = optimset('Display', 'off', 'TolX', 1e-8);
% --- Optimal Lambdas for Hybrid AB-GMRES ---
gcv_handle_ab = @(lambda) gcv_function(lambda, A, B_pert, b, m, k_gcv, 'ab');
[lambda_gcv_ab, ~] = fminbnd(gcv_handle_ab, 1e-9, 1e-1, options);
[min_err_ab, idx_true_opt_ab] = min(err_norms_ab);
lambda_true_optimal_ab = lambda_range(idx_true_opt_ab);
fprintf('   - Hybrid AB -> GCV Optimal: %.4e, True Optimal: %.4e\n', lambda_gcv_ab, lambda_true_optimal_ab);
% --- Optimal Lambdas for Hybrid BA-GMRES ---
gcv_handle_ba = @(lambda) gcv_function(lambda, A, B_pert, b, m, k_gcv, 'ba');
[lambda_gcv_ba, ~] = fminbnd(gcv_handle_ba, 1e-9, 1e-1, options);
[min_err_ba, idx_true_opt_ba] = min(err_norms_ba);
lambda_true_optimal_ba = lambda_range(idx_true_opt_ba);
fprintf('   - Hybrid BA -> GCV Optimal: %.4e, True Optimal: %.4e\n', lambda_gcv_ba, lambda_true_optimal_ba);
%% 4) Generate Figure 1: Analysis for Hybrid AB-GMRES
fprintf('4. Generating Figure 1: Analysis for Hybrid AB-GMRES...\n');
figure('Name', 'Regularization Analysis (Hybrid AB-GMRES)', 'Position', [100 100 1100 500]);
% --- L-Curve Plot for AB ---
subplot(1, 2, 1);
loglog(res_norms_ab, sol_norms_ab, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'L-Curve');
hold on;
[~, idx_gcv_ab] = min(abs(lambda_range - lambda_gcv_ab));
plot(res_norms_ab(idx_gcv_ab), sol_norms_ab(idx_gcv_ab), 'r*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'GCV Optimal');
plot(res_norms_ab(idx_true_opt_ab), sol_norms_ab(idx_true_opt_ab), 'gp', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'DisplayName', 'True Optimal');
hold off; grid on;
xlabel('Relative Residual Norm ||b - Ax_{\lambda}|| / ||b||');
ylabel('Solution Norm ||x_{\lambda}||');
title('L-Curve (Hybrid AB-GMRES)');
legend('Location', 'SouthEast');
set(gca, 'FontSize', 12);
% --- Error vs. Lambda Plot for AB ---
subplot(1, 2, 2);
loglog(lambda_range, err_norms_ab, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'Error Curve');
hold on;
plot(lambda_gcv_ab, err_norms_ab(idx_gcv_ab), 'r*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'GCV Optimal');
plot(lambda_true_optimal_ab, min_err_ab, 'gp', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'DisplayName', 'True Optimal');
hold off; grid on;
xlabel('Regularization Parameter \lambda');
ylabel('Relative Error Norm ||x_{\lambda} - x_{true}|| / ||x_{true}||');
title('Error vs. Lambda (Hybrid AB-GMRES)');
legend('Location', 'Best');
set(gca, 'FontSize', 12);
%% 5) Generate Figure 2: Analysis for Hybrid BA-GMRES
fprintf('5. Generating Figure 2: Analysis for Hybrid BA-GMRES...\n');
figure('Name', 'Regularization Analysis (Hybrid BA-GMRES)', 'Position', [150 150 1100 500]);
% --- L-Curve Plot for BA ---
subplot(1, 2, 1);
loglog(res_norms_ba, sol_norms_ba, 'm-x', 'LineWidth', 1.5, 'DisplayName', 'L-Curve');
hold on;
[~, idx_gcv_ba] = min(abs(lambda_range - lambda_gcv_ba));
plot(res_norms_ba(idx_gcv_ba), sol_norms_ba(idx_gcv_ba), 'r*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'GCV Optimal');
plot(res_norms_ba(idx_true_opt_ba), sol_norms_ba(idx_true_opt_ba), 'gp', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'DisplayName', 'True Optimal');
hold off; grid on;
xlabel('Relative Residual Norm ||b - Ax_{\lambda}|| / ||b||');
ylabel('Solution Norm ||x_{\lambda}||');
title('L-Curve (Hybrid BA-GMRES)');
legend('Location', 'SouthEast');
set(gca, 'FontSize', 12);
% --- Error vs. Lambda Plot for BA ---
subplot(1, 2, 2);
loglog(lambda_range, err_norms_ba, 'm-x', 'LineWidth', 1.5, 'DisplayName', 'Error Curve');
hold on;
plot(lambda_gcv_ba, err_norms_ba(idx_gcv_ba), 'r*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'GCV Optimal');
plot(lambda_true_optimal_ba, min_err_ba, 'gp', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'DisplayName', 'True Optimal');
hold off; grid on;
xlabel('Regularization Parameter \lambda');
ylabel('Relative Error Norm ||x_{\lambda} - x_{true}|| / ||x_{true}||');
title('Error vs. Lambda (Hybrid BA-GMRES)');
legend('Location', 'Best');
set(gca, 'FontSize', 12);
%% 6) Generate Figure 3: Visual Comparison of Final Hybrid Solutions
fprintf('6. Generating Figure 3: Visual Solution Comparison (Hybrid Methods)...\n');
% --- Solve for each hybrid method using its GCV-optimal lambda ---
x_optimal_ab = ABgmres_hybrid_bounds(A, B_pert, b, x_true, tol, maxit, lambda_gcv_ab, DeltaM_AB);
x_optimal_ba = BAgmres_hybrid_bounds(A, B_pert, b, x_true, tol, maxit, lambda_gcv_ba, DeltaM_BA);
% --- Create the plot ---
figure('Name', 'Visual Comparison of Hybrid Solutions', 'Position', [200 200 800 600]);
plot(1:n, x_true, 'k-', 'LineWidth', 3, 'DisplayName', 'True Solution');
hold on;
plot(1:n, x_optimal_ab, 'b--', 'LineWidth', 2, 'DisplayName', sprintf('Hybrid AB (\\lambda = %.2e)', lambda_gcv_ab));
plot(1:n, x_optimal_ba, 'm-.', 'LineWidth', 2, 'DisplayName', sprintf('Hybrid BA (\\lambda = %.2e)', lambda_gcv_ba));
hold off;
grid on;
title('Comparison of Final Solutions from Hybrid Methods');
xlabel('Element Index');
ylabel('Value');
legend('show', 'Location', 'northwest');
axis tight;
set(gca, 'FontSize', 12);
%% 7) Generate Figure 4: Visual Comparison of All Four Methods
fprintf('7. Generating Figure 4: Final Solution Comparison (All Methods)...\n');
% --- Solve with non-hybrid methods ---
solution_nonhybrid_ab = ABgmres_nonhybrid_bounds(A, B_pert, b, x_true, tol, maxit, DeltaM_AB);
solution_nonhybrid_ba = BAgmres_nonhybrid_bounds(A, B_pert, b, x_true, tol, maxit, DeltaM_BA);
% --- Solve with hybrid methods using their GCV-optimal lambda (re-using from above) ---
solution_hybrid_ab = x_optimal_ab;
solution_hybrid_ba = x_optimal_ba;
% --- Create the plot ---
figure('Name', 'Final Solution Comparison (All Methods)', 'Position', [250 250, 900, 650]);
plot(1:n, x_true, 'k-', 'LineWidth', 3.5, 'DisplayName', 'True Solution');
hold on;
plot(1:n, solution_nonhybrid_ab, ':', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2, 'DisplayName', 'non-hybrid AB (\lambda=0)');
plot(1:n, solution_nonhybrid_ba, ':', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'DisplayName', 'non-hybrid BA (\lambda=0)');
plot(1:n, solution_hybrid_ab, '-', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2, 'DisplayName', sprintf('hybrid AB (\\lambda=%.1e)', lambda_gcv_ab));
plot(1:n, solution_hybrid_ba, '-', 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2, 'DisplayName', sprintf('hybrid BA (\\lambda=%.1e)', lambda_gcv_ba));
hold off;
grid on;
title('Comparison of Final Solutions from All Methods');
xlabel('Element Index');
ylabel('Value');
legend('show', 'Location', 'northwest');
axis tight;
set(gca, 'FontSize', 12);
fprintf('--- All plotting complete. ---\n');
end
