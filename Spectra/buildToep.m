function T = buildToep(p, k)
%
%  Build a banded Toeplitz matrix from a central column and an index
%  denoting the central column.
%
n = length(p);
col = zeros(n,1);
row = col';
col(1:n-k+1,1) = p(k:n);
row(1,1:k) = p(k:-1:1)';
T = toeplitz(col, row);
