function run_2D_phantom()

clear all; clc; close all;
fprintf('Starting Final Thesis Experiments...\n\n');

n         = 32;          
noise_lvl = 0.1;         
maxit     = 34;          
lambda    = 1e-2;        
tol       = 1e-6;        

%% 2. GENERATE THE 2D TOMOGRAPHY PROBLEM 

fprintf('Generating %d x %d mismatched tomography problem...\n', n, n);
 
options.CTtype = 'fancurved';
[Problem, b_exact, x_true] = PRtomo_mismatched(n, options);
B = Problem.B;
A = Problem.A; 
 
rng(0);  
e = randn(size(b_exact));
e = e / norm(e) * noise_lvl * norm(b_exact);
b_noise = b_exact + e;

%% FIGURE 1: SINOGRAM

num_angles = 90;
num_detectors = 90;

% Check if our numbers are correct
assert(numel(b_exact) == num_angles * num_detectors, ...
    'Sinogram dimensions are incorrect!');

% Reshape the data vectors into 2D sinograms
sino_exact = reshape(b_exact, num_detectors, num_angles);
sino_noisy = reshape(b_noise, num_detectors, num_angles);

figure('Name', 'Figure 1: Sinogram Data');
sgtitle('Sinogram of the Shepp-Logan Phantom', 'FontSize', 14, 'FontWeight', 'bold');

% Plot the clean sinogram
subplot(1, 2, 1);
imagesc(sino_exact);
colormap gray;
axis xy; 
xlabel('Projection Index (k)', 'FontSize', 12);
ylabel('Detector Element', 'FontSize', 12);
title('a) Clean Sinogram (b_{exact})', 'FontSize', 12);
colorbar;

% Plot the noisy sinogram
subplot(1, 2, 2);
imagesc(sino_noisy);
colormap gray;
axis xy;
xlabel('Projection Index (k)', 'FontSize', 12);
title(sprintf('b) Noisy Sinogram (%.0f%% noise)', noise_lvl*100), 'FontSize', 12);
colorbar;

%% RUN SOLVERS

fprintf('Running solvers for initial reconstructions...\n');

% Run all four solvers using the initial mismatched B from the generator
[x_nonhy_ab, err_nonhy_ab, it_nonhy_ab] = gmres_nonhybrid_simple(A, B, b_noise, x_true, tol, maxit, 'AB');
[x_nonhy_ba, err_nonhy_ba, it_nonhy_ba] = gmres_nonhybrid_simple(A, B, b_noise, x_true, tol, maxit, 'BA');
[x_hy_ab, err_hy_ab, it_hy_ab] = gmres_hybrid_simple(A, B, b_noise, x_true, tol, maxit, lambda, 'AB');
[x_hy_ba, err_hy_ba, it_hy_ba] = gmres_hybrid_simple(A, B, b_noise, x_true, tol, maxit, lambda, 'BA');

fprintf('Initial solvers finished.\n\n');

fprintf('Generating figures...\n');


%% FIGURE: Reconstruction Quality Comparison

figure('Name', 'Figure 1: 2D Reconstruction Quality Comparison');
sgtitle('Comparison of 2D Reconstruction Methods', 'FontSize', 14, 'FontWeight', 'bold');
subplot(2, 2, 1); imagesc(reshape(x_nonhy_ab, n, n)); colormap gray; axis image; axis off; title('a) Non-Hybrid AB-GMRES', 'FontSize', 12);
subplot(2, 2, 2); imagesc(reshape(x_nonhy_ba, n, n)); colormap gray; axis image; axis off; title('b) Non-Hybrid BA-GMRES', 'FontSize', 12);
subplot(2, 2, 3); imagesc(reshape(x_hy_ab, n, n)); colormap gray; axis image; axis off; title(sprintf('c) Hybrid AB-GMRES'), 'FontSize', 12);
subplot(2, 2, 4); imagesc(reshape(x_hy_ba, n, n)); colormap gray; axis image; axis off; title(sprintf('d) Hybrid BA-GMRES'), 'FontSize', 12);

%% FIGURE: Semi-Convergence and Regularization Effect

figure('Name', 'Figure 2: Semi-Convergence');
semilogy(1:it_nonhy_ab, err_nonhy_ab, '--', 'LineWidth', 2, 'DisplayName', 'Non-Hybrid AB');
hold on;
semilogy(1:it_nonhy_ba, err_nonhy_ba, ':', 'LineWidth', 2, 'DisplayName', 'Non-Hybrid BA');
semilogy(1:it_hy_ab, err_hy_ab, '-', 'LineWidth', 2, 'DisplayName', 'Hybrid AB');
semilogy(1:it_hy_ba, err_hy_ba, '-.', 'LineWidth', 2, 'DisplayName', 'Hybrid BA');
grid on;
title('Semi-Convergence Behavior of GMRES Variants', 'FontSize', 14);
xlabel('Iteration (k)', 'FontSize', 12);
ylabel('Relative Error ||x_k - x_{true}|| / ||x_{true}||', 'FontSize', 12);
legend('show', 'Location', 'best');
hold off
%% FIGURE: Robustness to Mismatch

figure('Name', 'Figure 3: Robustness to Mismatch');
mismatch_levels = logspace(-4, 0, 10);
errors_hy_ab = zeros(size(mismatch_levels));
errors_hy_ba = zeros(size(mismatch_levels));
errors_nonhy_ab = zeros(size(mismatch_levels));
errors_nonhy_ba = zeros(size(mismatch_levels));

fprintf('Running mismatch robustness test...\n');
h = waitbar(0, 'Testing robustness to mismatch...');
for i = 1:length(mismatch_levels)
    E = randn(size(A'));
    E = E / norm(E, 'fro') * mismatch_levels(i);
    B_pert = A' + E;
    
    [~, err_nh_ab, ~] = gmres_nonhybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, 'AB');
    [~, err_nh_ba, ~] = gmres_nonhybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, 'BA');
    [~, err_h_ab, ~] = gmres_hybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, lambda, 'AB');
    [~, err_h_ba, ~] = gmres_hybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, lambda, 'BA');
    
    errors_nonhy_ab(i) = err_nh_ab(end);
    errors_nonhy_ba(i) = err_nh_ba(end);
    errors_hy_ab(i) = err_h_ab(end);
    errors_hy_ba(i) = err_h_ba(end);
    
    waitbar(i/length(mismatch_levels), h);
end
close(h);

loglog(mismatch_levels, errors_nonhy_ab, '--o', 'LineWidth', 2, 'DisplayName', 'Non-Hybrid AB');
hold on;
loglog(mismatch_levels, errors_nonhy_ba, ':s', 'LineWidth', 2, 'DisplayName', 'Non-Hybrid BA');
loglog(mismatch_levels, errors_hy_ab, '-o', 'LineWidth', 2, 'DisplayName', 'Hybrid AB');
loglog(mismatch_levels, errors_hy_ba, '-s', 'LineWidth', 2, 'DisplayName', 'Hybrid BA');
grid on;
title('Final Error vs. Back-Projector Mismatch', 'FontSize', 14);
xlabel('Mismatch Norm ||B - A^T||_F', 'FontSize', 12);
ylabel('Final Relative Error', 'FontSize', 12);
legend('show', 'Location', 'best');
 


%% HELPER FUNCTIONS
function [x, error_norm, niters] = gmres_nonhybrid_simple(A, B, b, x_true, tol, maxit, method_type)
 
    if strcmp(method_type, 'AB')
        M = A * B; 
        [z, ~, ~, iter_tuple] = gmres(M, b, [], tol, maxit);
        x = B * z;
        niters = iter_tuple(2);
 
        error_norm = zeros(niters, 1);
        if niters > 0
            for k=1:niters 
                z_k = gmres(M, b, [], tol, k); 
                x_k = B * z_k;
                error_norm(k) = norm(x_k - x_true) / norm(x_true);
            end
        end 

    else % BA
        M = B * A;
        d = B * b; 
        [x, ~, ~, iter_tuple] = gmres(M, d, [], tol, maxit);
        niters = iter_tuple(2);
 
        error_norm = zeros(niters, 1);
        if niters > 0
            for k=1:niters 
                x_k = gmres(M, d, [], tol, k);
                error_norm(k) = norm(x_k - x_true) / norm(x_true);
            end
        end 
    end
end

function [x, error_norm, niters] = gmres_hybrid_simple(A, B, b, x_true, tol, maxit, lambda, method_type)
 
    if strcmp(method_type, 'AB')
        M = A * B;
        [z, ~, ~, niters] = lsqr([M; sqrt(lambda)*eye(size(M,2))], [b; zeros(size(M,2),1)], tol, maxit);
        x = B * z; 
        error_norm = zeros(niters, 1);
        if niters > 0
            for k=1:niters
                [z_k] = lsqr([M; sqrt(lambda)*eye(size(M,2))], [b; zeros(size(M,2),1)], tol, k);
                x_k = B * z_k;
                error_norm(k) = norm(x_k - x_true) / norm(x_true);
            end
        end
    else % BA
        M = B * A;
        d = B * b;
        [x, ~, ~, niters] = lsqr([M; sqrt(lambda)*eye(size(M,1))], [d; zeros(size(M,1),1)], tol, maxit);
        error_norm = zeros(niters, 1);
        if niters > 0
            for k=1:niters
                [x_k] = lsqr([M; sqrt(lambda)*eye(size(M,1))], [d; zeros(size(M,1),1)], tol, k);
                error_norm(k) = norm(x_k - x_true) / norm(x_true);
            end
        end
    end
end
end