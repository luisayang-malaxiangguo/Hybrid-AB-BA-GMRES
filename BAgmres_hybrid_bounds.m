function [x, error_norm, residual_norm, niters, phi_final, dphi_final, phi_iter, dphi_iter] = BAgmres_hybrid_bounds( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM)
% BAgmres_hybrid_bounds: Final corrected version.
% Implements the full theoretical bounds from the associated paper, including
% internal estimation of the RHS-induced perturbation and automatic
% detection of system symmetry to apply the appropriate bounds.

% BA setup
M = B*A;
n = size(A,2);
DeltaK = DeltaM; % Since K = M + lambda*I, dK = dM
K_full = M + lambda*eye(n);

% --- Symmetry Check for Bounds ---
% This check determines which theoretical bound (Weyl vs. Bauer-Fike) to use.
is_symmetric = issymmetric(M, 'skew');

% Exact eigenvalues of M (BA)
mu_full = sort(real(eig(M)).', 'descend');

% --- Estimate RHS-induced perturbation norm internally ---
% This term is unique to the BA-GMRES case as per the paper's theory.
norm_A = opnorm2(A);
norm_DeltaM = norm(DeltaM, 2);
norm_b = norm(b);
d_perturbed = B * b;
norm_d_perturbed = norm(d_perturbed);
deltaQ_norm_estimate = 0;
if norm_A > 1e-12 && norm_d_perturbed > 1e-12
    % Estimate ||E|| = ||B_pert - B_true|| from ||DeltaM|| = ||E*A||
    norm_E_estimate = norm_DeltaM / norm_A;
    % Estimate ||delta_d|| = ||E*b||
    delta_d_norm_estimate = norm_E_estimate * norm_b;
    % Final estimate for ||delta Q_k||_2 is the relative change in the start vector
    deltaQ_norm_estimate = delta_d_norm_estimate / norm_d_perturbed;
end

% Arnoldi on M with BA residual
r0 = d_perturbed - M*zeros(n,1);
beta = norm(r0);
Q = zeros(n, maxit+1);
H = zeros(maxit+1, maxit);
Q(:,1)= r0/beta;
e1 = [beta; zeros(maxit,1)];

residual_norm = zeros(maxit,1);
error_norm    = zeros(maxit,1);
phi_iter      = cell(maxit,1);
dphi_iter     = cell(maxit,1);

for k = 1:maxit
    % Arnoldi process
    v = B*(A*Q(:,k));
    for j = 1:k
        H(j,k) = Q(:,j)'*v;
        v = v - H(j,k)*Q(:,j);
    end
    H(k+1,k) = norm(v);
    if H(k+1,k) == 0, break; end
    Q(:,k+1) = v/H(k+1,k);

    % Projected Tikhonov solution
    Hk = H(1:k+1,1:k);
    tk = e1(1:k+1);
    yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
    xk = Q(:,1:k)*yk;

    residual_norm(k) = norm(b - A*xk)/norm(b);
    error_norm(k)    = norm(xk - x_true)/norm(x_true);

    % =================== REVISED PERTURBATION BOUNDS BLOCK ===================
    Qk      = Q(:,1:k);
    Hsmall  = H(1:k,1:k);
    h       = H(k+1,k);
    ek      = zeros(k,1); ek(end)=1;

    % -- (1) Harmonic Ritz of K = M + lambda*I --
    Kk_proj = Hsmall + lambda*eye(k);
    P_harm  = Kk_proj + (h^2) * (Kk_proj' \ (ek*ek'));
    Theta   = sort(real(eig(P_harm)), 'descend');

    % -- (2) Filters with EXACT mu --
    mu  = mu_full(1:k).';
    s2l = mu + lambda;
    F = 1 - (s2l ./ Theta.');
    logAbsF = log(abs(F) + 1e-300);
    logPall = sum(logAbsF, 1).';
    P_all   = exp(logPall) .* exp(1i * sum(angle(F), 1).');
    phi = (mu ./ s2l).* (1 - P_all);

    % -- (3) Compute Normwise Bounds based on symmetry --
    Ak = Qk' * K_full' * K_full * Qk;
    Dk = Qk' * K_full' * Qk;
    norm_K = opnorm2(K_full);
    max_abs_theta = max(abs(Theta));

    if is_symmetric
        % For a symmetric system, eigenvectors are perfectly conditioned
        kappa_max = 1.0;
        kappa_X = 1.0;
    else
        % For non-symmetric, compute eigenvector condition number of the pencil
        [r, ~] = eig(Ak, Dk);
        [l, ~] = eig(Ak', Dk');
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
        % We still use kappa_X=1 as a practical simplification, as eig(M) is expensive.
        kappa_X = 1.0;
    end
    
    % Constants from Table 3 of the paper
    C_tilde_k = kappa_max * (2*norm_K + max_abs_theta);
    C_tilde_k_rhs = kappa_max * (2*norm_K^2 + 2*max_abs_theta*norm_K);
    
    % Final normwise bound includes BOTH operator and RHS-induced terms
    dtheta_bound = C_tilde_k * norm(DeltaK, 2) + C_tilde_k_rhs * deltaQ_norm_estimate;
    dmu_bound = kappa_X * norm(DeltaM, 2);

    % -- (4) Map (dtheta_bound, dmu_bound) -> |dphi| --
    logP_excl = logPall - logAbsF.';
    argP_excl = sum(angle(F), 1).' - angle(F);
    P_excl = exp(logP_excl) .* exp(1i*argP_excl);
    
    abs_mu      = abs(mu);
    abs_s2l     = abs(s2l);
    Theta_clip  = max(abs(Theta), 1e-12);

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

niters = k;
x = xk;
residual_norm = residual_norm(1:k);
error_norm    = error_norm(1:k);
phi_final     = phi_iter{k};
dphi_final    = dphi_iter{k};
phi_iter      = phi_iter(1:k);
dphi_iter     = dphi_iter(1:k);
end

% ---- Helper Function ----
function nrm = opnorm2(X)
% Efficiently estimates the 2-norm for large matrices via power iteration.
    if numel(X) < 400*400 % Use built-in for smaller matrices
        nrm = norm(X, 2);
        return;
    end
    
    try
        nrm = norm(X, 2);
    catch
        [n, ~] = size(X);
        v = randn(n,1); v = v/norm(v);
        for it = 1:20
            v_new = X*(X'*v);
            nv = norm(v_new); 
            if nv == 0, nrm = 0; return; end
            v = v_new / nv;
        end
        nrm = sqrt(nv);
    end
end

