num=1;
Fs=8000;

Root='.\';
TrainDataRoot=[Root 'database\train\'];
TrainFeatureRoot=[Root 'features\train\'];
TestDataRoot=[Root 'database\test\SNR5_male\'];

TrainFeatureRoot=[Root 'features\train\'];
TestFeatureRoot=[Root 'features\test\'];

trainfiles=dir(TrainDataRoot);
trainfiles=trainfiles(3:end);
testfiles=dir(TestDataRoot);
testfiles=testfiles(3:end);
features=dir(TestFeatureRoot);

file_name=char(testfiles(num).name);
%file_name=char(trainfiles(num).name);
wavFile=[TestDataRoot file_name];
rawdata=open_wavfile(wavFile);


winlen      = 25;            % window length in 100 nsec
winshft     = 10;
% medlen      = 100;
% medshft     = 50;
nfft        = 1024;            % fft size 
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
% medlen = round(medlen * 10^(-3) * fsamp);
% medshft = medshft * 10^(-3) * fsamp;
if isempty(whos('FrameNo' ))
    FrameNo = ceil((length(input) - winlen) / winshft);
end

% MedNo = ceil((length(input) - medlen) / medshft);

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

% do a similar thing with the inputs on a medium time basis. 
% medwin = sig2fm(preem, medlen, medshft, MedNo);
% [medlen, medframenum] = size(medwin);

% Hamming window each frame
frmwin = frmwin .* (hamming(winlen) * ones(1, framenum));
%medwin = medwin .* (hamming(medlen) * ones(1, medframenum));

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

