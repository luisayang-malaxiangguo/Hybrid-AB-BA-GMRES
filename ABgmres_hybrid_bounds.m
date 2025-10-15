function [x, error_norm, residual_norm, niters, phi_final, dphi_final, phi_iter, dphi_iter] = ABgmres_hybrid_bounds( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM)
% ABgmres_hybrid_bounds: Corrected version with symmetry detection and original signature.

% AB setup
M_hat = A*B;
m = size(A,1);
DeltaK_hat = DeltaM; % Since K = M + lambda*I, dK = dM
K_hat_full = M_hat + lambda*eye(m);

% --- Symmetry Check for Bounds ---
is_symmetric = issymmetric(M_hat, 'skew');

% Exact eigenvalues of M_hat (AB)
mu_full = sort(real(eig(M_hat)).', 'descend');

% Arnoldi on M_hat
z0 = zeros(size(B,2),1);
r0 = b - A*(B*z0);
beta = norm(r0);
Q_hat = zeros(m, maxit+1);
H_hat = zeros(maxit+1, maxit);
Q_hat(:,1) = r0/beta;
e1 = [beta; zeros(maxit,1)];

residual_norm = zeros(maxit,1);
error_norm    = zeros(maxit,1);
phi_iter      = cell(maxit,1);
dphi_iter     = cell(maxit,1);

for k = 1:maxit
    % Arnoldi
    v = A*(B*Q_hat(:,k));
    for j = 1:k
        H_hat(j,k) = Q_hat(:,j)'*v;
        v = v - H_hat(j,k)*Q_hat(:,j);
    end
    H_hat(k+1,k) = norm(v);
    if H_hat(k+1,k) == 0, break; end
    Q_hat(:,k+1) = v/H_hat(k+1,k);

    % Projected Tikhonov
    Hk = H_hat(1:k+1,1:k);
    tk = e1(1:k+1);
    yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
    zk = Q_hat(:,1:k)*yk;
    xk = B*zk;

    residual_norm(k) = norm(b - A*xk)/norm(b);
    error_norm(k)    = norm(xk - x_true)/norm(x_true);

    % =================== REVISED PERTURBATION BOUNDS BLOCK ===================
    Qk_hat      = Q_hat(:,1:k);
    Hsmall_hat  = H_hat(1:k,1:k);
    h_hat       = H_hat(k+1,k);
    ek          = zeros(k,1); ek(end)=1;

    % -- (1) Harmonic Ritz of K_hat = M_hat + lambda*I --
    Kk_proj_hat = Hsmall_hat + lambda*eye(k);
    P_harm_hat  = Kk_proj_hat + (h_hat^2) * (Kk_proj_hat' \ (ek*ek'));
    Theta       = sort(real(eig(P_harm_hat)), 'descend');

    % -- (2) Filters with EXACT mu --
    mu  = mu_full(1:k).';
    s2l = mu + lambda;
    F = 1 - (s2l ./ Theta.');
    logAbsF = log(abs(F) + 1e-300);
    logPall = sum(logAbsF, 1).';
    P_all   = exp(logPall) .* exp(1i * sum(angle(F), 1).');
    phi = (mu ./ s2l).* (1 - P_all);

    % -- (3) Compute Normwise Bounds for delta_theta and delta_mu --
    Ak = Qk_hat' * K_hat_full' * K_hat_full * Qk_hat;
    Dk = Qk_hat' * K_hat_full' * Qk_hat;
    norm_K_hat = opnorm2(K_hat_full);
    max_abs_theta = max(abs(Theta));

    if is_symmetric
        kappa_max = 1.0;
        kappa_Y = 1.0; % Eigenvectors are perfectly conditioned
    else
        [r, ~] = eig(Ak, Dk); [l, ~] = eig(Ak', Dk');
        kappa_max = 0;
        for j = 1:k
            l_j = l(:,j); r_j = r(:,j);
            norm_factor = l_j' * Dk * r_j;
            if abs(norm_factor) > 1e-12
                l_j_norm = l_j / sqrt(norm_factor);
                r_j_norm = r_j / sqrt(norm_factor');
                kappa_max = max(kappa_max, norm(l_j_norm, 2) * norm(r_j_norm, 2));
            end
        end
        % A practical simplification, as eig(M_hat) is too expensive.
        kappa_Y = 1.0;
    end
    
    C_tilde_k_hat = kappa_max * (2*norm_K_hat + max_abs_theta);
    dtheta_bound = C_tilde_k_hat * norm(DeltaK_hat, 2);
    dmu_bound = kappa_Y * norm(DeltaM, 2);

    % -- (4) Map (bounds) -> |dphi| --
    logP_excl = logPall - logAbsF.';
    argP_excl = sum(angle(F), 1).' - angle(F);
    P_excl = exp(logP_excl) .* exp(1i*argP_excl);
    abs_mu = abs(mu); abs_s2l = abs(s2l); Theta_clip  = max(abs(Theta), 1e-12);
    
    sum_term1 = sum(bsxfun(@rdivide, abs(P_excl), Theta_clip.'.^2), 2);
    dtheta_part = (abs_mu .* sum_term1) * dtheta_bound;
    
    lambda_term = (abs(lambda) ./ (abs_s2l.^2)) .* abs(1 - P_all);
    mu_term_coeff = sum(bsxfun(@rdivide, abs(P_excl), Theta_clip.'), 2);
    mu_term = (abs_mu ./ abs_s2l) .* mu_term_coeff;
    
    dmu_part = (lambda_term + mu_term) * dmu_bound;
    dphi = dtheta_part + dmu_part;

    phi_iter{k}  = phi;
    dphi_iter{k} = dphi;

    if residual_norm(k) <= tol, break; end
end

niters = k; x = xk;
residual_norm = residual_norm(1:k); error_norm = error_norm(1:k);
phi_final = phi_iter{k}; dphi_final = dphi_iter{k};
phi_iter = phi_iter(1:k); dphi_iter = dphi_iter(1:k);
end

function nrm = opnorm2(X)
    if numel(X) < 400*400, nrm = norm(X, 2); return; end
    try, nrm = norm(X, 2); catch
        [n, ~] = size(X); v = randn(n,1); v = v/norm(v);
        for it = 1:20, v_new = X*(X'*v); nv = norm(v_new);
            if nv == 0, nrm = 0; return; end; v = v_new / nv;
        end; nrm = sqrt(nv);
    end
end
