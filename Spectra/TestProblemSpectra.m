function [A, b, x_true] = TestProblemSpectra
%
% This function will generage a test problem similar to image deblurring,
% but it is a small 1-dimensional signal deblurring problem. 
% The matrix is very ill-conditioned, so we may need to incorporate
% regularization, especially if we add noise to the vector b.
%
x_true = spectra2;
[PSF, center] = psfGauss(size(x_true));
%[PSF, center] = psfGauss(size(x_true), 1); 
A = buildToep(PSF, center);
b_true = A*x_true;
b = PRnoise(b_true, 0); % the value 0 means add no noise, so really 
                        % b = b_true. But I put it here in case we want
                        % to add noise later (change 0 to, say, 0.001).
                        % See the doc PRnoise for more information.