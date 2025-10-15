function [x, error_norm, residual_norm, niters] = hybrid_ab_gmres_rtp(A, B, b, x_true, tol, maxit, lambda)
% HYBRID_AB_GMRES_RTP  Regularize-then-Project (RTP) version of hybrid AB-GMRES.
% Build GMRES on (AB + λI) z = b in R^m, then map back x = B z.
% No projected Tikhonov here (that would be PTR).

    m = size(A,1);
    z0 = zeros(m,1);

    % Shifted AB operator
    Mreg = @(v) A*(B*v) + lambda*v;

    % Arnoldi setup in R^m
    r0 = b - Mreg(z0);            % with z0=0, r0 = b
    beta = norm(r0);
    Q = zeros(m, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0 / max(beta, eps);

    error_norm    = zeros(maxit,1);
    residual_norm = zeros(maxit,1);
    x = B*z0;

    for k = 1:maxit
        % Arnoldi step
        v = Mreg(Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)'*v;
            v = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0, break; end
        Q(:,k+1) = v / H(k+1,k);

        % GMRES subproblem on shifted system (NO Tikhonov here)
        Hk = H(1:k+1, 1:k);
        yk = Hk \ ([beta; zeros(k,1)]);     % least squares (overdetermined)

        zk = Q(:,1:k) * yk;                 % z_k in R^m
        x  = B * zk;                         % AB mapping

        residual_norm(k) = norm(b - A*x) / max(norm(b), eps);
        error_norm(k)    = norm(x - x_true) / max(norm(x_true), eps);

        if residual_norm(k) <= tol, break; end
    end

    niters        = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
end
