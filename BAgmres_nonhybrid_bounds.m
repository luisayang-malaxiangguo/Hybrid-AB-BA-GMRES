function [x, err, res, niters, phi_final, dphi_final, phi_iter, dphi_iter] = BAgmres_nonhybrid_bounds( ...
    A, B, b, x_true, tol, maxit, DeltaM)
% BAGMRES_NONHYBRID_BOUNDS Non-hybrid BA-GMRES + perturbation bounds.
% return filter factors AND their perturbation bounds at each iteration.
    
    % Build M = B*A
    M = B * A;
    n = size(A, 2);

    % The eigenvalues 'mu' must be computed from the actual system matrix M = B*A.
    % Since M may be non-symmetric, we use eig().
    [V_M, D_M] = eig(M);
    mu_full = real(diag(D_M));
    % Sorting is crucial for consistent mode analysis
    [mu_full, sort_idx] = sort(mu_full, 'descend');
    % The matrix VA used later should be the eigenvectors of M
    VA = V_M(:, sort_idx);

    %--- Arnoldi GMRES on M x = B*b ---%
    d0    = B * b;
    r0    = d0;
    beta  = norm(r0);
    Q     = zeros(n, maxit+1);
    H     = zeros(maxit+1, maxit);
    Q(:,1)= r0 / beta;
    res   = zeros(maxit,1);
    err   = zeros(maxit,1);
    
    % Initialize cell arrays to store results
    phi_iter = cell(maxit, 1);
    dphi_iter = cell(maxit, 1);
    
    for k = 1:maxit
        % Arnoldi step
        v = M * Q(:,k);
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v      = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k)==0, break; end
        Q(:,k+1) = v / H(k+1,k);
        
        % Solve Hk * y = beta * e1
        Hk  = H(1:k+1, 1:k);
        yk  = Hk \ ([beta; zeros(k,1)]);
        xk  = Q(:,1:k) * yk;
        
        % Norms in original space
        res(k)  = norm(b - A * xk) / norm(b);
        err(k)  = norm(xk - x_true) / norm(x_true);
        
        
        % Compute harmonic-Ritz values and their perturbations
        Qk_current = Q(:,1:k);
        Hk_small_current = H(1:k,1:k);
        dKk_current = Qk_current' * DeltaM * Qk_current;
        ek_current = zeros(k,1); 
        ek_current(end) = 1;
        P_eig_problem = Hk_small_current + (H(k+1,k)^2) * (Hk_small_current'\(ek_current*ek_current'));
        [W_current, Th_eig] = eig(P_eig_problem);
        Theta_current = real(diag(Th_eig));
        [Theta_current, p_sort] = sort(Theta_current);
        W_current = W_current(:, p_sort);
        
        % Perturbation of harmonic-Ritz values
        dTheta_current = real(diag(W_current' * dKk_current * W_current));
        
        % CORRECTED perturbation of eigenvalues using eigenvectors of M
        dMu_current = sum(VA(:,1:k) .* (DeltaM * VA(:,1:k)), 1)';
        
        % Truncate mu to the current size k
        mu_current = mu_full(1:k); % This now uses the CORRECT eigenvalues
        
        % Build phi and dphi using filter factor formula from Theorem 4.2.1
        eps0_current = eps;
        Clog_current = zeros(k,1);
        P_excl_current = zeros(k,k);
        for i = 1:k
            factors = max(1 - mu_current(i)./Theta_current.', eps0_current);
            Clog_current(i) = sum(log(factors));
            for j = 1:k
                denom = max(1 - mu_current(i)/Theta_current(j), eps0_current);
                P_excl_current(i,j) = exp(Clog_current(i) - log(denom));
            end
        end
        P_final = exp(Clog_current);
        phi_current = 1 - P_final;
        
        % First-order perturbation bound for non-hybrid method
        term1 = - mu_current .* sum((dTheta_current' ./ Theta_current'.^2) .* P_excl_current, 2);
        term2 =  sum((1./Theta_current') .* P_excl_current, 2) .* dMu_current;
        dphi_current = term1 + term2;
        
        % Store the calculated phi and dphi in our cell arrays
        phi_iter{k} = phi_current;
        dphi_iter{k} = dphi_current;
        
        if res(k) <= tol, break; end
    end
    niters = k;
    x      = xk;
    res    = res(1:k);
    err    = err(1:k);

    phi_final = phi_iter{k};
    dphi_final = dphi_iter{k};
    
    phi_iter = phi_iter(1:k);
    dphi_iter = dphi_iter(1:k);
end