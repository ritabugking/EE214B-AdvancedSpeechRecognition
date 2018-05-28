function feature = PNCC(rawdata,Fs)

winlen      = 25;            % window length in 100 nsec
if nargin == 3
    winshft = 1000/120;      
else
    winshft = 10;            % window shift in 100 nsec %samples
end
nfft        = 512;            % fft size 
cepnum      = 12;             % number of cepstral coefficients
liftercoe   = 22;             % liftering coefficient
numchan     = 26;             % number of channels of the MEL filter bank 26
preemcoeff  = 0.97;           % coefficient for pre-emphasis
deltawindow = 2;              % window length to calculate 1st derivatives
accwindow   = 2;              % window length to calculate 2nd derivatives
C0          = 1;              % use zeroth cepstral coefficient (0/1)

% -------------------------------------------------------------
% START OF PROGRAM
input = rawdata;
fsamp = Fs;
winlen = round(winlen * 10^(-3) * fsamp);
winshft = winshft * 10^(-3) * fsamp;
if isempty(whos('FrameNo' ))
    FrameNo = ceil((length(input) - winlen) / winshft);
end

% initialize MEL filter bank
fbank = initfiltb(winlen, numchan, fsamp, nfft);

% initialize lifter coefficients
lifter = (1 + (liftercoe/2)*sin((pi/liftercoe)*(0:cepnum)) );

% pre-emphasis
am = [1 0];  % denominator polynomial
bm = [1 - preemcoeff];            % numerator   polynomial
preem = filter(bm, am, input);

% change signal (a vector) into frame (a matrix), where each column is a frame
frmwin = sig2fm(preem, winlen, winshft, FrameNo);      
[winlen, framenum] = size(frmwin); 

% Hamming window each frame
frmwin = frmwin .* (hamming(winlen) * ones(1, framenum));


%=====================================================================
%================= Start of Noise Robust Processing ==================
%=====================================================================


% FFT
ffto = abs(fft(frmwin, nfft));

% MEL filtering (currently replacing the gammatone)
fb = fbank*ffto(1 : (nfft/2), :);
p = fbank.^2 * ffto(1:(nfft/2),:).^2;

mpmat = movavg(p, 2); % M = 2 according to paper. can change. 
q = p * mpmat;

% figure
% hold on
% for i=1:size(mp, 1)
%     %subplot(size(mp, 2), i, 1)
%     plot(mp(i,:));
% end

qle = anr_filter(q, 0.999, 0.5); % can change these values. 
                                    % 1 > b > a > 0 means upper envelope. 
                                    % 1 > a > b > 0 means lower envelope. 
                                    

q0 = q - qle;        % noise removal. 
q0(q0<0) = 0;           % half wave rectification. 

% following portion deals with zero intervals and small local variances.
qf = anr_filter(q0, 0.999, 0.5); 
qtm = temporal_mask(q0, 0.85, 0.2); % temporal masking (can change vals)
rsp = max(qtm, qf);

excite = q >= 2*qle;
r = excite.*rsp + ~excite.*qf;
s = spectral_smoothing(q, r, 4);
t = p.*s;

mu = mean_power(t, 0.999); % not sure why it's 0.999 but it's apparently data dependent.
% might need VAD as well. 
k = 1; % supposedly this is arbitrary. 
u = k * t ./ mu'; 
v = u .^ (1/15); % there's also an approximation. 

%=====================================================================
%================= End of Noise Robust Processing ====================
%=====================================================================



% calculate 1st derivative (velocity)
dt1 = deltacc(v, deltawindow);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(dt1, accwindow);
% append dt1 and dt2 to mfcco
v = [v; dt1; dt2];
feature = v';

% END OF PROGRAM
% ---------------------------------------------------------------


