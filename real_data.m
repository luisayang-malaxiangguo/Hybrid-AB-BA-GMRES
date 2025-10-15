%% ——— Replace PRtomo with your real data ———

%— Step A: Inspect what’s actually inside the .mat file—
gtFile = 'GroundTruthReconstruction.mat';
vars = whos('-file', gtFile);
if isempty(vars)
    error('No variables found inside %s', gtFile);
end
fprintf('Found variable(s): %s\n', strjoin({vars.name}, ', '));

%— Step B: Load the first variable it contains and vectorize it—
S      = load(gtFile, vars(1).name);
x_true = S.(vars(1).name)(:);

%% START FROM HERE

%% ——— Complete 256×256 workflow with robust sparse thresholding ———

% 1) Load your down-sampled measurement data
load('DataFull_256x45.mat','A','m','normA');
b = m(:);   % vectorize sinogram

% 2) Load the FBP ground truth (512×512) and down-sample it
S_full = load('GroundTruthReconstruction.mat');   % contains FBP360
vars   = fieldnames(S_full);
GT512  = S_full.(vars{1});                        
GT256  = imresize(GT512, [256,256]);              
x_true = GT256(:);

% 3) Build ProbInfo for reshaping & plotting
N              = 256;                             
ProbInfo.xSize = [N, N];
ProbInfo.bSize = size(m);                         

% 4) (Optional) Add 3% noise to b
[bn,NoiseInfo] = PRnoise(b, 0.03);

% 5) Build approximate back-projector B with threshold τ *safely*
tau = 0.01;
T   = tau * max(abs(A(:)));

% Instead of B = A.'; B(abs(B)<T)=0; which densifies the matrix,
% do this on the sparse structure directly:
[i,j,v] = find(A.');               % get nonzeros of A'
keep    = abs(v) >= T;             % logical mask
B       = sparse(i(keep), j(keep), v(keep), size(A,2), size(A,1));

% 6) Visualize sinogram and ground truth
figure(1), imagesc(reshape(b,ProbInfo.bSize)),   axis image off, colormap gray
title('Sinogram b (256×45)')
figure(2), imagesc(reshape(x_true,ProbInfo.xSize)), axis image off, colormap gray
title('Ground Truth x\_true (256×256)')

% 7) Example solver: Tikhonov‐regularized PCG
alpha = 0.1 * normA^2;                             
fun   = @(x) A.'*(A*x) + alpha*x;
rhs   = A.' * bn;                                  
tol   = 1e-6; maxit = 100;
[x_rec,flag] = pcg(fun, rhs, tol, maxit);
if flag~=0
    warning('PCG did not converge fully (flag=%d)', flag);
end

% 8) Display reconstruction
img_rec = reshape(x_rec, N, N);
figure(3), imagesc(img_rec), axis image off, colormap gray
title('Tikhonov‐PCG Reconstruction (256×256)')


%% ——— Add noise (if you like) ———
[bn, NoiseInfo] = PRnoise(b, 0.03);
rel_noise = norm(NoiseInfo.noise)/norm(b);


%% ——— Now call your solvers exactly as before ———

tol   = 1e-6;
maxit = 100;
lambda = 100;

% Given A is m×n, we want B ≈ A' but zero out small entries
tau = 0.01;
T   = tau * max(abs(A(:)));

% 1) transpose sparsely
spAt = A.';  

% 2) define a function that zeroes out small values
threshFcn = @(x) x .* (abs(x) >= T);

% 3) apply it only to the nonzeros
B = spfun(threshFcn, spAt);  

%%
%% Precompute once: nonzeros of A'
[i0, j0, v0] = find(A.');     % A.' is sparse, so this is cheap

tau_list = [0.01, 0.1, 0.3, 0.5];
leg = cell(size(tau_list));

% Prepare figure
figure;
for p = 1:4
    subplot(2,2,p);
    hold on; grid on;
end

%% 1) AB-GMRES
for idx = 1:numel(tau_list)
    tau = tau_list(idx);
    T   = tau * max(abs(v0));       % threshold in value space
    
    keep = abs(v0) >= T;            % logical mask
    Btmp = sparse(i0(keep), j0(keep), v0(keep), size(A,2), size(A,1));

    [~, errAB, ~, ~] = ABgmres_own(A, Btmp, bn, x_true, tol, maxit);
    subplot(2,2,1);
    plot(errAB,'-o');
    leg{idx} = sprintf('\\tau = %.2f', tau);
end
title('AB-GMRES','Interpreter','tex');

%% 2) BA-GMRES
for idx = 1:numel(tau_list)
    tau = tau_list(idx);
    T   = tau * max(abs(v0)); 
    keep = abs(v0) >= T;
    Btmp = sparse(i0(keep), j0(keep), v0(keep), size(A,2), size(A,1));

    [~, errBA, ~, ~] = BAgmres_own(A, Btmp, bn, x_true, tol, maxit);
    subplot(2,2,2);
    plot(errBA,'-s');
end
title('BA-GMRES','Interpreter','tex');

%% 3) Hybrid AB-GMRES
for idx = 1:numel(tau_list)
    tau = tau_list(idx);
    T   = tau * max(abs(v0)); 
    keep = abs(v0) >= T;
    Btmp = sparse(i0(keep), j0(keep), v0(keep), size(A,2), size(A,1));

    [~, errABhy, ~, ~] = ABgmres_hybrid(A, Btmp, bn, x_true, tol, maxit, lambda);
    subplot(2,2,3);
    plot(errABhy,'-d');
end
title('Hybrid AB-GMRES','Interpreter','tex');

%% 4) Hybrid BA-GMRES
for idx = 1:numel(tau_list)
    tau = tau_list(idx);
    T   = tau * max(abs(v0)); 
    keep = abs(v0) >= T;
    Btmp = sparse(i0(keep), j0(keep), v0(keep), size(A,2), size(A,1));

    [~, errBAhy, ~, ~] = BAgmres_hybrid(A, Btmp, bn, x_true, tol, maxit, lambda);
    subplot(2,2,4);
    plot(errBAhy,'-^');
end
title('Hybrid BA-GMRES','Interpreter','tex');

% Add legends and labels
for p = 1:4
    subplot(2,2,p);
    xlabel('Iteration k'); ylabel('Relative error');
    legend(leg,'Interpreter','tex','Location','northeast');
end






