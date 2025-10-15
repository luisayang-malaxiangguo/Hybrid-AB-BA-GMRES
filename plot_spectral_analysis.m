function plot_spectral_analysis()
% This script generates a 2x2 plot showing the true eigenvalues of the
% system matrix and ritz/harmonic ritz values
clear all;
clc;

%% 1) Set up Test Problem and True Eigenvalues
fprintf('1. Setting up the test problem and computing true eigenvalues...\n');
n = 32;
problem_name = 'deriv2';
[A, b_exact, ~] = generate_test_problem(problem_name, n);

% For this analysis, we use a slightly perturbed back-projector
rng(0);
E = 1e-4 * randn(size(A'));
B_pert = A' + E;
 
M_ab = A * B_pert;
mu_ab_true = sort(real(eig(M_ab)), 'ascend');

M_ba = B_pert * A;
mu_ba_true = sort(real(eig(M_ba)), 'ascend');

%% 2) Generate the 2x2 Plot
fprintf('2. Generating spectral approximation plots...\n');
k_values = [5, 15, 30]; % Iterations to visualize
lambda = 1e-3;  

figure('Name', 'Spectral Approximation Analysis', 'Position', [100 100 900 750]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Harmonic Ritz Values (\theta) vs. True Eigenvalues (\mu)', 'FontSize', 14, 'FontWeight', 'bold');

%  Subplot for non-hybrid AB-GMRES (uses Ritz Values) 
ax1 = nexttile;
plot_single_method_spectrum(ax1, 'nonhybrid_ab', A, B_pert, b_exact, mu_ab_true, k_values, lambda);
title('non-hybrid AB-GMRES');
ylabel('Value (log scale)');

%  Subplot for non-hybrid BA-GMRES (uses Harmonic Ritz Values) 
ax2 = nexttile;
plot_single_method_spectrum(ax2, 'nonhybrid_ba', A, B_pert, b_exact, mu_ba_true, k_values, lambda);
title('non-hybrid BA-GMRES');

%  Subplot for hybrid AB-GMRES (uses Harmonic Ritz of regularized op) 
ax3 = nexttile;
plot_single_method_spectrum(ax3, 'hybrid_ab', A, B_pert, b_exact, mu_ab_true, k_values, lambda);
title('hybrid AB-GMRES');
xlabel('Eigenvalue Index');
ylabel('Value (log scale)');

%  Subplot for hybrid BA-GMRES (uses Harmonic Ritz of regularized op) 
ax4 = nexttile;
plot_single_method_spectrum(ax4, 'hybrid_ba', A, B_pert, b_exact, mu_ba_true, k_values, lambda);
title('hybrid BA-GMRES');
xlabel('Eigenvalue Index');

fprintf(' Analysis complete. \n');
end

%% Helper function to plot the spectrum for a single method
function plot_single_method_spectrum(ax, method_type, A, B, b, mu_true, k_vals, lambda)
    hold(ax, 'on');
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
    
    % Plot true eigenvalues as black dots
    semilogy(ax, 1:length(mu_true), mu_true, 'k.', 'MarkerSize', 12, 'DisplayName', 'True \mu_i');
    
    % For each k, compute and plot the spectral approximations
    for i = 1:length(k_vals)
        k = k_vals(i);
        theta_k = get_spectral_approximations(method_type, A, B, b, k, lambda);
        
        % Plot the computed thetas
        if ~isempty(theta_k)
            semilogy(ax, 1:length(theta_k), theta_k, 'o', 'MarkerSize', 7, ...
                     'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), ...
                     'DisplayName', sprintf('\\theta_j for k = %d', k));
        end
    end
    
    hold(ax, 'off');
    grid on;
    legend('Location', 'SouthEast');
    xlim([0, length(mu_true) + 1]);
    set(gca, 'FontSize', 11);
end

%% Helper function to run Arnoldi and compute spectral approximations
function theta = get_spectral_approximations(method_type, A, B, b, k_target, lambda)

    % Choose operator and starting vector
    if contains(method_type, 'ab')
        op = @(v) A * (B * v);  % M̂ = AB (m×m)
        r0 = b;                 % size m
        op_size = size(A,1);
    else % 'ba'
        op = @(v) B * (A * v);  % M = BA (n×n)
        r0 = B * b;             % size n
        op_size = size(A,2);
    end

    % Arnoldi
    Q = zeros(op_size, k_target+1);
    H = zeros(k_target+1, k_target);
    beta = norm(r0);
    if beta == 0, theta = []; return; end
    Q(:,1) = r0 / beta;

    k_actual = 0;
    for k = 1:k_target
        v = op(Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12, k_actual = k; break; end
        Q(:,k+1) = v / H(k+1,k);
        k_actual = k;
    end

    Hk = H(1:k_actual, 1:k_actual);
    hk1k = H(k_actual+1, k_actual);
    ek = zeros(k_actual,1); ek(end) = 1;

    % Build the (A, B) pencil for harmonic Ritz
    switch method_type
        case {'nonhybrid_ab','nonhybrid_ba'}
            % Harmonic Ritz of M (λ = 0)
            Apen = (Hk'*Hk) + (abs(hk1k)^2) * (ek*ek.');
            Bpen = Hk';
        case {'hybrid_ab','hybrid_ba'}
            % Harmonic Ritz of K = M + λ I
            Apen = (Hk'*Hk) ...
                 + lambda*(Hk' + Hk) ...
                 + (lambda^2)*eye(k_actual) ...
                 + (abs(hk1k)^2) * (ek*ek.');
            Bpen = Hk' + lambda*eye(k_actual);
        otherwise
            error('Unknown method_type');
    end

    % Solve generalized eigenproblem Apen * y = theta * Bpen * y
    theta_all = eig(Apen, Bpen);
    theta = sort(real(theta_all),'ascend');
end
