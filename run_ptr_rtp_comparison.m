function run_ptr_rtp_comparison()
% Plot PTR and RTP iterate vectors (component vs index) at k = 32
% for BA- and AB-hybrid methods. Uses only your existing solvers.

clear; clc; close all;

%% 1) Test problem (n = 32 so k=32 is the cap)
fprintf('Setting up test problem...\n');
n = 64;                 % <- target iteration
kplot = n;             % which iterate to plot
[A, b_exact, x_true] = generate_test_problem('deriv2', n);
B = A';                 % matched-transpose case

rng(0);
noise   = randn(size(b_exact));
b_noise = b_exact + 1e-2 * norm(b_exact) * noise / norm(noise);

lambda = 1e-3;

%% 2) Compute PTR and RTP iterates at k = 32 (disable early stopping)
fprintf('Computing iterates at k = %d...\n', kplot);

% --- BA pair ---
[x_ba_ptr, ~, ~, it_ba_ptr] = BAgmres_hybrid_bounds(A,B,b_noise,x_true,0,kplot,lambda,zeros(size(B*A)));
[x_ba_rtp, ~, ~, it_ba_rtp] = hybrid_ba_gmres_rtp    (A,B,b_noise,x_true,0,kplot,lambda);

% --- AB pair ---
[x_ab_ptr, ~, ~, it_ab_ptr] = ABgmres_hybrid_bounds(A,B,b_noise,x_true,0,kplot,lambda,zeros(size(A*B)));
[x_ab_rtp, ~, ~, it_ab_rtp] = hybrid_ab_gmres_rtp    (A,B,b_noise,x_true,0,kplot,lambda);

% Warn if any method broke down before k=32
if it_ba_ptr < kplot || it_ba_rtp < kplot || it_ab_ptr < kplot || it_ab_rtp < kplot
    warning('Some method(s) terminated before k = %d. Plots still use their returned iterate.', kplot);
end

%% 3) Plots: PTR vs RTP iterate vectors at k = 32
fprintf('Generating plots...\n');
figure('Name', sprintf('PTR vs RTP', kplot), 'Position', [180, 180, 1100, 520]);

% BA
subplot(1,2,1);
plot(1:n, x_ba_ptr, 'b-',  'LineWidth', 2, 'DisplayName', 'BA-PTR'); hold on;
plot(1:n, x_ba_rtp, 'm-.', 'LineWidth', 2, 'DisplayName', 'BA-RTP');
plot(1:n, x_true,   'k:',  'LineWidth', 1.5, 'DisplayName', 'x_{true}'); hold off;
grid on; xlabel('Component index k'); ylabel('x_k');
title(sprintf('BA-GMRES: PTR vs RTP', kplot));
legend('show','Location','best');

% AB
subplot(1,2,2);
plot(1:n, x_ab_ptr, 'b-',  'LineWidth', 2, 'DisplayName', 'AB-PTR'); hold on;
plot(1:n, x_ab_rtp, 'm-.', 'LineWidth', 2, 'DisplayName', 'AB-RTP');
plot(1:n, x_true,   'k:',  'LineWidth', 1.5, 'DisplayName', 'x_{true}'); hold off;
grid on; xlabel('Component index k'); ylabel('x_k');
title(sprintf('AB-GMRES: PTR vs RTP', kplot));
legend('show','Location','best');

sgtitle(sprintf('Validation of PTR RTP Inequivalence', kplot), 'FontSize', 16, 'FontWeight', 'bold');


end
