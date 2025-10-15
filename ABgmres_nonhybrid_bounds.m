function [x, err, res, niters, phi_final, dphi_final, phi_iter, dphi_iter] = ABgmres_nonhybrid_bounds( ...
    A, B, b, x_true, tol, maxit, DeltaM)
    
    % Build M = A*B and get its eigenvalues/vectors for analysis
    M = A * B;
    m = size(A, 1);
    
    % The eigenvalues 'mu' mre computed from the actual system matrix M = A*B.
    [U_M, D_M] = eig(M);
    mu_full = real(diag(D_M));
    % Sorting for consistent mode analysis
    [mu_full, sort_idx] = sort(mu_full, 'descend');
    % matrix UA stores the eigenvectors of M, needed for dMu calculation
    UA = U_M(:, sort_idx);

    %--- Arnoldi GMRES on M z = b ---%    
    z0    = zeros(size(B,2),1);
    r0    = b - A*(B*z0);
    beta  = norm(r0);
    Q     = zeros(m, maxit+1);
    H     = zeros(maxit+1, maxit);
    Q(:,1)= r0 / beta;
    res   = zeros(maxit,1);
    err   = zeros(maxit,1);
    
    % Initialize to store filter factors and perturbation bounds
    phi_iter = cell(maxit, 1);
    dphi_iter = cell(maxit, 1);

    for k = 1:maxit
        % Arnoldi step
        v = A*(B*Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v      = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k)==0, break; end
        Q(:,k+1) = v / H(k+1,k);
        
        % Solve projected least-squares Hk*y = beta*e1
        Hk = H(1:k+1,1:k);
        yk = Hk \ ([beta; zeros(k,1)]);
        z  = Q(:,1:k) * yk;
        xk = B * z;
        
        % Compute norms & check convergence
        res(k) = norm(b - A*xk)/norm(b);
        err(k) = norm(xk - x_true)/norm(x_true);
        
        
        % Compute harmonic-Ritz values and their perturbations
        Qk_current = Q(:,1:k);
        Hk_small_current = H(1:k,1:k);
        % The projected perturbation is Qk'*DeltaM*Qk
        dKk_current = Qk_current' * DeltaM * Qk_current;
        ek_current = zeros(k,1); 
        ek_current(end) = 1;
        
        % computes the harmonic-Ritz values of M
        P_eig_problem = Hk_small_current + (H(k+1,k)^2)*(Hk_small_current'\(ek_current*ek_current'));
        [W_current, Th_eig] = eig(P_eig_problem);
        Theta_current = real(diag(Th_eig));
        [Theta_current, p_sort] = sort(Theta_current); % sort ascending
        W_current = W_current(:, p_sort);   % reorder harmonic-Ritz vectors
        
        % Perturbation of harmonic-Ritz values
        dTheta_current = real(diag(W_current' * dKk_current * W_current));
        
        % CORRECTED perturbation of eigenvalues using eigenvectors of M=A*B
        dMu_current = sum(UA(:,1:k) .* (DeltaM * UA(:,1:k)), 1)';
        
        % Truncate mu to the current size k
        mu_current = mu_full(1:k);
        
        % Build phi and dphi using formula for non-hybrid methods
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
        phi_current = 1 - P_final; % Filter factors for this k
        
        % Perturbation bound calculation for non-hybrid method (two terms)
        term1 = - mu_current .* sum((dTheta_current' ./ Theta_current'.^2) .* P_excl_current, 2);
        term2 =  sum((1./Theta_current') .* P_excl_current, 2) .* dMu_current;
        dphi_current = term1 + term2; % Perturbation bound for this k
        
        % Store results
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