function plot_perturbation_bound_validation() 
% Validates the theoretical perturbation bounds by comparing them against
% the actual change in filter factors (magnitudes).

clc; clear; close all

%% 1) Test problem & params
n = 32;
problem_name = 'heat';                       
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

lambda = 1e-3;                               % regularization
maxit  = n;
tol    = 1e-6;

rng(0)                                       % reproducible
B_unpert = A';
E = 1e-4 * randn(size(A));
B_pert   = B_unpert + E';

DeltaM_AB = A * E';                           % AB: Δ(A*B) = A*E'
DeltaM_BA = E' * A;                           % BA: Δ(B*A) = E'*A

%% 2) Run simulations
fprintf('Running simulations for k <= %d...\n', maxit);

% --- hybrid BA-GMRES ---
[~,~,~,~,~,~, phi_hba_u, dphi_hba_bound] = BAgmres_hybrid_bounds(A, B_unpert, b_exact, x_true, tol, maxit, lambda, DeltaM_BA);
[~,~,~,~,~,~, phi_hba_p, ~]              = BAgmres_hybrid_bounds(A, B_pert,   b_exact, x_true, tol, maxit, lambda, zeros(size(DeltaM_BA)));

% --- hybrid AB-GMRES ---
[~,~,~,~,~,~, phi_hab_u, dphi_hab_bound] = ABgmres_hybrid_bounds(A, B_unpert, b_exact, x_true, tol, maxit, lambda, DeltaM_AB);
[~,~,~,~,~,~, phi_hab_p, ~]              = ABgmres_hybrid_bounds(A, B_pert,   b_exact, x_true, tol, maxit, lambda, zeros(size(DeltaM_AB)));

fprintf('Simulations complete.\n');

%% 3) Plot comparison (magnitudes)
fprintf('Generating bound validation plot...\n');
figure('Name', 'Perturbation Bound Validation', 'Position', [100 100 900 700]);
t = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Validation of Perturbation Bounds at Final Iteration', 'FontSize', 14, 'FontWeight', 'bold');

plot_single_bound(nexttile, phi_hab_u, phi_hab_p, dphi_hab_bound, 'hybrid AB-GMRES');
plot_single_bound(nexttile, phi_hba_u, phi_hba_p, dphi_hba_bound, 'hybrid BA-GMRES');

end

function plot_single_bound(ax, phi_u, phi_p, dphi_bound, plot_title)
    k_cells = min([length(phi_u), length(phi_p), length(dphi_bound)]);
    if k_cells == 0
        title(ax, [plot_title ' (No iterations completed)']);
        return;
    end

    % magnitudes
    phi_p_k      = phi_p{k_cells};
    phi_u_k      = phi_u{k_cells};
    dphi_bound_k = dphi_bound{k_cells};

    actual_change     = abs(phi_p_k - phi_u_k);
    theoretical_bound = abs(dphi_bound_k);

    m = numel(actual_change);
    semilogy(ax, 1:m, actual_change, 'o-',  'DisplayName','Actual Change |Δφ|'); hold(ax, 'on');
    semilogy(ax, 1:m, theoretical_bound, 'x--','DisplayName','Theoretical Bound |δφ|'); hold(ax, 'off');
    grid(ax, 'on');
    title(ax, sprintf('%s (k=%d)', plot_title, k_cells));
    xlabel(ax, 'Mode index i');
    ylabel(ax, 'Magnitude');
    legend(ax, 'Location','Best');
end