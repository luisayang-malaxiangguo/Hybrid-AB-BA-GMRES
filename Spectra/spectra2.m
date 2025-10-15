%
%  f=spectra2;
%
%  This subroutine constructs a simulated gamma
%  ray spectra. See Trussell, IEEE Trans. on ASSP,
%  V31 (1983), pp. 129--136.
%
function f=spectra2

f=zeros(64,1);
f(21,1)=7;
f(22,1)=14;
f(23,1)=7;

f(26,1)=4;
f(27,1)=8;
f(28,1)=4;

f(35,1)=3;
f(36,1)=6;
f(37,1)=3;

f(43,1)=1;

