function [x, error_norm, residual_norm, niters] = hybrid_ba_gmres_rtp(A, B, b, x_true, tol, maxit, lambda)
% HYBRID_BA_GMRES_RTP  Regularize-then-Project (RTP) version of hybrid BA-GMRES.
% Build GMRES on (BA + λI) x = Bb in R^n. No projected Tikhonov here.

    n = size(A,2);
    x0 = zeros(n,1);

    % Shifted BA operator
    Mreg = @(v) B*(A*v) + lambda*v;
    d    = B*b;

    % Arnoldi setup in R^n
    r0 = d - Mreg(x0);            % with x0=0, r0 = d
    beta = norm(r0);
    Q = zeros(n, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0 / max(beta, eps);

    error_norm    = zeros(maxit,1);
    residual_norm = zeros(maxit,1);
    x = x0;

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
        yk = Hk \ ([beta; zeros(k,1)]);     % least squares

        x  = Q(:,1:k) * yk;                 % BA maps directly

        residual_norm(k) = norm(b - A*x) / max(norm(b), eps);
        error_norm(k)    = norm(x - x_true) / max(norm(x_true), eps);

        if residual_norm(k) <= tol, break; end
    end

    niters        = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
end
