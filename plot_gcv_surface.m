function plot_gcv_surface()
% PLOT_GCV_SURFACE Generates a 2D color map of the GCV
% function value vs. (iteration, lambda) for BOTH hybrid methods.
% It produces a separate figure for each method for clarity.

clear all;
clc;

%% 1) Set up Test Problem and Parameters
fprintf('1. Setting up the test problem...\n');
n = 32;
problem_name = 'shaw';
[A, b_exact, ~] = generate_test_problem(problem_name, n);

% --- Add Noise and Perturbation ---
rng(0); % For reproducibility
noise_level = 1e-2;
noise = randn(size(b_exact));
b_noise = b_exact + (noise / norm(noise)) * noise_level * norm(b_exact);

E = 1e-4 * randn(size(A'));
B_pert = A' + E;

% --- Define the Grid for the Surface Plot ---
k_range = 1:n;
lambda_range = logspace(-8, -1, 100);

%% 2) Compute GCV Surface for Hybrid AB-GMRES
fprintf('2. Computing GCV surface for Hybrid AB-GMRES...\n');
[gcv_surface_ab, gcv_path_ab] = compute_gcv_surface('ab', A, B_pert, b_noise, n, k_range, lambda_range);

%% 3) Compute GCV Surface for Hybrid BA-GMRES
fprintf('3. Computing GCV surface for Hybrid BA-GMRES...\n');
[gcv_surface_ba, gcv_path_ba] = compute_gcv_surface('ba', A, B_pert, b_noise, n, k_range, lambda_range);

%% 4) Generate Plot for Hybrid AB-GMRES
fprintf('4. Generating plot for Hybrid AB-GMRES...\n');
figure('Name', 'GCV Surface (Hybrid AB-GMRES)', 'Position', [100 100 800 650]);
imagesc(k_range, lambda_range, log10(gcv_surface_ab));
hold on;
plot(k_range, gcv_path_ab, 'r-p', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'GCV Optimal \lambda_k');
hold off;
set(gca, 'YDir', 'normal', 'YScale', 'log');
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'log_{10}(GCV Value)');
xlabel('Iteration Count (k)');
ylabel('Regularization Parameter (\lambda)');
title('GCV Surface for Hybrid AB-GMRES');
legend('show', 'Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 12);

%% 5) Generate Plot for Hybrid BA-GMRES
fprintf('5. Generating plot for Hybrid BA-GMRES...\n');
figure('Name', 'GCV Surface (Hybrid BA-GMRES)', 'Position', [950 100 800 650]);
imagesc(k_range, lambda_range, log10(gcv_surface_ba));
hold on;
plot(k_range, gcv_path_ba, 'm-p', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', 'GCV Optimal \lambda_k');
hold off;
set(gca, 'YDir', 'normal', 'YScale', 'log');
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'log_{10}(GCV Value)');
xlabel('Iteration Count (k)');
ylabel('Regularization Parameter (\lambda)');
title('GCV Surface for Hybrid BA-GMRES');
legend('show', 'Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 12);

fprintf('--- Analysis complete. ---\n');
end


% --- HELPER: Main function to compute a GCV surface ---
function [gcv_surface, gcv_path] = compute_gcv_surface(method_type, A, B, b, n, k_range, lambda_range)
    % Initialize storage
    gcv_surface = zeros(length(lambda_range), length(k_range));
    gcv_path = zeros(length(k_range), 1);

    % Set up operator based on method type
    if strcmp(method_type, 'ab')
        op = @(v) A * (B * v);
        r0 = b;
        op_size = size(A, 1);
    else % 'ba'
        op = @(v) B * (A * v);
        r0 = B * b;
        op_size = size(A, 2);
    end

    % --- Perform Arnoldi Process Once ---
    Q = zeros(op_size, n + 1);
    H = zeros(n + 1, n);
    beta = norm(r0);
    Q(:,1) = r0 / beta;
    e1_base = [beta; zeros(n, 1)];

    for k = k_range
        % --- Single Arnoldi Step ---
        v = op(Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12, H(k+1:end, k:end) = 0; break; end
        Q(:,k+1) = v / H(k+1,k);
        
        % --- For this k, calculate GCV for all lambdas ---
        Hk = H(1:k+1, 1:k);
        tk = e1_base(1:k+1);
        gcv_values_for_k = zeros(length(lambda_range), 1);
        
        for i = 1:length(lambda_range)
            lambda = lambda_range(i);
            gcv_values_for_k(i) = calculate_gcv_from_H(Hk, tk, lambda, op_size);
        end
        
        gcv_surface(:, k) = gcv_values_for_k;
        
        % Find and store the best lambda for this k
        [~, min_idx] = min(gcv_values_for_k);
        gcv_path(k) = lambda_range(min_idx);
    end
end


% --- HELPER: GCV calculation from a pre-computed Hessenberg matrix ---
function gcv_val = calculate_gcv_from_H(Hk, tk, lambda, problem_size)
    k = size(Hk, 2);
    
    % Solve the Tikhonov projected problem
    yk = (Hk' * Hk + lambda * eye(k)) \ (Hk' * tk);
    
    % Numerator of the GCV function
    residual_norm_sq = norm(tk - Hk * yk)^2;
    
    % Denominator of the GCV function (trace term)
    [~, S, ~] = svd(Hk(1:k, 1:k), 'econ');
    s_diag = diag(S);
    
    trace_val = sum(s_diag.^2 ./ (s_diag.^2 + lambda));
    denominator = (problem_size - trace_val)^2;
    
    gcv_val = residual_norm_sq / denominator;
    
    if isnan(gcv_val) || isinf(gcv_val) || denominator < eps
        gcv_val = 1e20; % Return a large value if unstable
    end
end
