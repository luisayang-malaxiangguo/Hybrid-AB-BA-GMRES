function gcv_val = gcv_function(lambda, A, B, b, m, k_gcv, gcv_type)
    
    % initial residual and Arnoldi step depend on the method type.
    if strcmp(gcv_type, 'ab')
        r0 = b;
        n_arnoldi = m; % Arnoldi is in m-space
    else % 'ba'
        r0 = B * b;
        n_arnoldi = size(A, 2); % Arnoldi is in n-space
    end
    
    beta = norm(r0);
    Q = zeros(n_arnoldi, k_gcv + 1);
    H = zeros(k_gcv + 1, k_gcv);
    Q(:,1) = r0 / beta;
    e1 = [beta; zeros(k_gcv, 1)];

    for k = 1:k_gcv
        if strcmp(gcv_type, 'ab')
            v = A * (B * Q(:,k));
        else % 'ba'
            v = B * (A * Q(:,k));
        end
        
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12, break; end
        Q(:,k+1) = v / H(k+1,k);
    end
    k = size(H, 2);
    
    Hk = H(1:k+1, 1:k);
    tk = e1(1:k+1);
    
    yk = (Hk' * Hk + lambda * eye(k)) \ (Hk' * tk);
    
    residual_norm_sq = norm(tk - Hk * yk)^2;
    
    [~, S, ~] = svd(H(1:k, 1:k), 'econ');
    s_diag = diag(S);
    
    % trace term depends on the method
    if strcmp(gcv_type, 'ab')
        trace_m = m;
    else % 'ba'
        trace_m = size(A,2);
    end
    trace_val = sum(s_diag.^2 ./ (s_diag.^2 + lambda));
    denominator = (trace_m - trace_val)^2;
    
    gcv_val = residual_norm_sq / denominator;
    
    if isnan(gcv_val) || isinf(gcv_val) || denominator < eps
        gcv_val = 1e20;
    end
end
