function hybrid_2D_only()

clear all; clc; close all;
fprintf('Starting Final Thesis Experiments...\n\n');

n         = 32;          
noise_lvl = 0.1;         
maxit     = 80;          
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


 
%% FIGURE: Robustness to Mismatch

figure('Name', 'Figure 3: Robustness to Mismatch');
mismatch_levels = logspace(-4, 0, 10);
errors_hy_ab = zeros(size(mismatch_levels));
errors_hy_ba = zeros(size(mismatch_levels)); 

fprintf('Running mismatch robustness test...\n');
h = waitbar(0, 'Testing robustness to mismatch...');
for i = 1:length(mismatch_levels)
    E = randn(size(A'));
    E = E / norm(E, 'fro') * mismatch_levels(i);
    B_pert = A' + E;
   [~, err_h_ab, ~] = gmres_hybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, lambda, 'AB');
    [~, err_h_ba, ~] = gmres_hybrid_simple(A, B_pert, b_noise, x_true, tol, maxit, lambda, 'BA');
     
    errors_hy_ab(i) = err_h_ab(end);
    errors_hy_ba(i) = err_h_ba(end);
    
    waitbar(i/length(mismatch_levels), h);
end
close(h);
loglog(mismatch_levels, errors_hy_ab, '-o', 'LineWidth', 2, 'DisplayName', 'Hybrid AB');
hold on;
loglog(mismatch_levels, errors_hy_ba, '-s', 'LineWidth', 2, 'DisplayName', 'Hybrid BA');
grid on;
title('Final Error vs. Back-Projector Mismatch', 'FontSize', 14);
xlabel('Mismatch Norm ||B - A^T||_F', 'FontSize', 12);
ylabel('Final Relative Error', 'FontSize', 12);
legend('show', 'Location', 'best');
 


%% HELPER FUNCTIONS

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