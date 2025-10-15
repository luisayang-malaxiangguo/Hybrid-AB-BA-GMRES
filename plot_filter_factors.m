function plot_filter_factors()
%close all;
%% 1) Set up Test Problem
n = 32;
problem_to_run = 'shaw';
[A, b_exact, x_true] = generate_test_problem(problem_to_run, n);

rng(0); % For reproducibility
noise_level = 1e-3; 
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b_noise = b_exact + noise; % Noisy right-hand side

%% 2) Algorithm Parameters
tol     = 1e-6;
maxit   = 32;     
lambda  = 1e-3;
B       = A';     % Matched back-projector
% Use a fixed seed for the random perturbation for reproducibility
rng(0); 
% Define perturbation E and perturbed matrix B
E = 1e-4 * randn(size(A'));
B_pert= B + E;

% Define the total perturbation terms for AB and BA methods
DeltaM_AB = A * E';
DeltaM_BA = E' * A;

%% 3) Run Each Method & Collect Full Iterative History
fprintf('Running GMRES variants...\n');

[x_ab, err_ab, res_ab, it_ab, phi_ab_final, dphi_ab_final, phi_ab_iter,  dphi_ab_iter] = ABgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_AB);
[x_ba, err_ba, res_ba, it_ba, phi_ba_final, dphi_ba_final, phi_ba_iter,  dphi_ba_iter] = BAgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_BA);
% If you want to keep using the noisy RHS everywhere:
[x_hab, err_hab, res_hab, it_hab, phi_hab_final, dphi_hab_final, phi_hab_iter, dphi_hab_iter] = ...
    ABgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(DeltaM_AB)));
[x_hba, err_hba, res_hba, it_hba, phi_hba_final, dphi_hba_final, phi_hba_iter, dphi_hba_iter] = ...
    BAgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(DeltaM_BA)));

fprintf('All methods complete.\n');

%% 4) Plot 1: Final Theoretical vs. Empirical Factors (2x2 Subplot)
fprintf('Generating final filter factor comparison plot...\n');

% Compute empirical filters  
[U,S,V] = svd(A,'econ');
sigma   = diag(S);
d       = U' * b_noise;

% Ensure division by d is safe for near-zero values
d(abs(d) < 1e-12) = 1; 

Phi_emp_ab  = sigma .* (V' * x_ab) ./ d;
Phi_emp_ba  = sigma .* (V' * x_ba) ./ d;
Phi_emp_hab = sigma .* (V' * x_hab) ./ d;
Phi_emp_hba = sigma .* (V' * x_hba) ./ d;

figure('Name', 'Final Filter Factor Comparison', 'Position', [100 100 800 600]);

% Subplot 1: non-hybrid AB
subplot(2,2,1);
kmin = min(numel(phi_ab_final), numel(Phi_emp_ab));
plot(1:kmin, real(phi_ab_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_ab(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('AB-GMRES (non-hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 2: non-hybrid BA
subplot(2,2,2);
kmin = min(numel(phi_ba_final), numel(Phi_emp_ba));
plot(1:kmin, real(phi_ba_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_ba(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('BA-GMRES (non-hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 3: hybrid AB
subplot(2,2,3);
kmin = min(numel(phi_hab_final), numel(Phi_emp_hab));
plot(1:kmin, real(phi_hab_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_hab(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('AB-GMRES (hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');

% Subplot 4: hybrid BA
subplot(2,2,4);
kmin = min(numel(phi_hba_final), numel(Phi_emp_hba));
plot(1:kmin, real(phi_hba_final(1:kmin)), '--', 'LineWidth', 1.6); hold on;
plot(1:kmin, real(Phi_emp_hba(1:kmin)), 'o-', 'MarkerSize', 4);
hold off; grid on;
title('BA-GMRES (hybrid)');
xlabel('Mode index i'); ylabel('Filter factor \phi_i');
legend('Theoretical', 'Empirical', 'Location', 'Best');



%% Plot 5: Unified Convergence History

figure('Name', 'Unified Convergence History', 'Position', [100 100 1000 400]);
lw = 1.8;

% Subplot for Relative Error
subplot(1, 2, 1);
semilogy(1:it_ab, err_ab, '--', 'LineWidth', lw, 'DisplayName', 'non-hybrid AB'); hold on;
semilogy(1:it_ba, err_ba, ':', 'LineWidth', lw, 'DisplayName', 'non-hybrid BA');
semilogy(1:it_hab, err_hab, '-', 'LineWidth', lw, 'DisplayName', 'hybrid AB');
semilogy(1:it_hba, err_hba, '-.', 'LineWidth', lw, 'DisplayName', 'hybrid BA');
hold off;
title('Relative Error vs. Iteration');
xlabel('Iteration k');
ylabel('||x_k - x_{true}|| / ||x_{true}||');
legend('Location', 'Best');
grid on;

% Subplot for Relative Residual
subplot(1, 2, 2);
semilogy(1:it_ab, res_ab, '--', 'LineWidth', lw, 'DisplayName', 'non-hybrid AB'); hold on;
semilogy(1:it_ba, res_ba, ':', 'LineWidth', lw, 'DisplayName', 'non-hybrid BA');
semilogy(1:it_hab, res_hab, '-', 'LineWidth', lw, 'DisplayName', 'hybrid AB');
semilogy(1:it_hba, res_hba, '-.', 'LineWidth', lw, 'DisplayName', 'hybrid BA');
hold off;
title('Relative Residual vs. Iteration');
xlabel('Iteration k');
ylabel('||b - Ax_k|| / ||b||');
legend('Location', 'Best');
grid on;

end