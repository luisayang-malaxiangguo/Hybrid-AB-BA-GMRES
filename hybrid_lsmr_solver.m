function [x, error_norm, residual_norm, niters] = hybrid_lsmr_solver(A, b, x_true, tol, maxit, lambda)
% HYBRID_LSMR_SOLVER Solves Tikhonov problem using Hybrid LSMR.

    n = size(A, 2);
    x = zeros(n, 1);
    
    u = b;
    beta1 = norm(u);
    u = u / beta1;
    
    V = zeros(n, maxit);
    B_k = zeros(maxit+1, maxit);
    
    v_hat = A' * u;
    alpha1 = norm(v_hat);
    v = v_hat / alpha1;
    V(:,1) = v;
    
    error_norm = zeros(maxit, 1);
    residual_norm = zeros(maxit, 1);
    
    for k = 1:maxit
        %  GKB  
        B_k(k,k) = alpha1;
        u_hat = A * v - alpha1 * u;
        beta_k = norm(u_hat);
        u = u_hat / beta_k;
        B_k(k+1, k) = beta_k;
        
        if k < maxit
            v_hat = A' * u - beta_k * v;
            alpha_k_plus_1 = norm(v_hat);
            v = v_hat / alpha_k_plus_1;
            V(:, k+1) = v;
            alpha1 = alpha_k_plus_1;
        end
         
        Bk = B_k(1:k+1, 1:k);
        alpha_k1 = alpha1;
        beta_k1 = beta_k;
        
        LHS = (Bk'*Bk)^2 + (alpha_k1*beta_k1)^2 * (eye(k,1)*eye(1,k)) + lambda * eye(k);
        RHS = B_k(1,1) * beta1 * (Bk'*Bk) * eye(k,1);
        
        yk = LHS \ RHS;
        x = V(:, 1:k) * yk;
 
        error_norm(k) = norm(x - x_true) / norm(x_true);
        residual_norm(k) = norm(b - A*x) / norm(b);
        
        if residual_norm(k) <= tol, break; end
    end
    
    niters = k;
    error_norm = error_norm(1:k);
    residual_norm = residual_norm(1:k);
end