%rawdata=template;
%Fs=8000;
function feature = PCA(rawdata,Fs,a)
alpha = a; % vtln warping factor
    winlen = 25;
    winshft =10;
    cepnum  =13;
    liftercoe =22;
    numchan =26;
    preemcoeff =0.97;
    deltawindow =2;
    accwindow =2;
    %C0 =0;
    usedeltas =1; 
    useenergy =0;
    
nfft        = 512;            % fft size 
cepnum      = 13;             % number of cepstral coefficients
liftercoe   = 22;             % liftering coefficient
numchan     = 26;             % number of channels of the MEL filter bank 26
preemcoeff  = 0.97;           % coefficient for pre-emphasis
deltawindow = 2;              % window length to calculate 1st derivatives
accwindow   = 2;              % window length to calculate 2nd derivatives
C0          = 1;              % use zeroth cepstral coefficient (0/1)
% window length in 100 nsec
% window shift in 100 nsec %samples
% number of cepstral coefficients
% liftering coefficient
% number of channels of the MEL filter bank 26 % coefficient for pre-emphasis
% window length to calculate 1st derivatives
% window length to calculate 2nd derivatives
% use zeroth cepstral coefficient (0/1)
% ------------------------------------------------------------- % START OF PROGRAM
input = rawdata;
fsamp = Fs;
winlen = round(winlen * 10^(-3) * fsamp); 
winshft = winshft * 10^(-3) * fsamp;
if isempty(whos('FrameNo' ))
    FrameNo = ceil((length(input) - winlen) / winshft);
end
nfft = 2*(2^nextpow2(winlen)); % FFT size 

% initialize MEL filter bank
fbank = initfiltb(winlen, numchan, fsamp, nfft);

% initialize lifter coefficients
lifter = (1 + (liftercoe/2)*sin((pi/liftercoe)*(0:cepnum-1)) );

% pre-emphasis
am = [1 0];  % denominator polynomial
bm = [1 - preemcoeff];            % numerator   polynomial
preem = filter(bm, am, input);

% change signal (a vector) into frame (a matrix), where each collum is a frame
frmwin = sig2fm(preem, winlen, winshft, FrameNo);      
[winlen, framenum] = size(frmwin); 

% Hamming window each frame
frmwin = frmwin .* (hamming(winlen) * ones(1, framenum));


% ============ LPCC =============
% Log energy
LE = log(sum(frmwin.*frmwin));
% FFT
fft_mag = zeros(cepnum, framenum); 
for i = 1:framenum
    lpcc = msf_lpcc(frmwin(:, i),8000, 'order',cepnum);
    fft_mag(:, i) =lpcc;
end
lpccs=fft_mag;

% calculate 1st derivative (velocity)
dt1 = deltacc(lpccs, deltawindow);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(dt1, accwindow);
% append dt1 and dt2 to mfcco
lpccs = [lpccs; dt1; dt2];
lpccs = lpccs';

% ============ MFCC ============
% % FFT
% ffto = abs(fft(frmwin, nfft));
% 
% % MEL filtering 
% fb = fbank*ffto(1 : (nfft/2), :);
% FFT
ffto = abs(fft(frmwin, nfft));

ffto = ffto(1 : (nfft/2), :);

%vtln
ffto2= ffto;
a=struct();
for i=1:size(ffto2,2)
a(i).data=ffto2(:,i);
end
aa=vtln(a, 'asymmetric', alpha);
ffto3=[];
for i=1:size(ffto2,2)
    ffto3=[ffto3, aa(i).data];
end

ffto = ffto3;







% MEL filtering 
%fb = fbank*ffto(1 : (nfft/2), :);
fb = fbank*ffto;

% take logarithm of MEL filter output
fbfloor = mean(mean(fb)) * 0.00001;  
logfb = log(max(fb, fbfloor*rand(size(fb))));

% take DCT
mfcco = dct(logfb);
if C0
    mfcco  = mfcco(1 : cepnum-1 + 1, :);
else
    mfcco  = mfcco(2 : cepnum-1 + 1, :);
    lifter = lifter(2 : end);
end

% do liftering with a lifted sin wave
mfcco = mfcco .* (lifter' * ones(1, framenum));

% cepstral mean subtraction (optional)
mfcco = mfcco - mean(mfcco, 2) * ones(1, size(mfcco, 2));

%=====================================================================
%================= End of Noise Robust Processing ====================
%=====================================================================

% calculate 1st derivative (velocity)
dt1 = deltacc(mfcco, deltawindow);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(dt1, accwindow);
% append dt1 and dt2 to mfcco
mfcco = [mfcco; dt1; dt2];
mfcco = mfcco';

%==============PLP=========================
data=rawdata;
[feature, spectra, pspectrum, lpcas, F, M] = rastaplp(data,Fs,0,12);
     % calculate 1st derivative (velocity)
dt1 = deltacc(feature, 2);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(dt1, 2);
% append dt1 and dt2 to mfcco
feature = [feature; dt1; dt2];
plps = feature';



% [W1, pc1] = pca(mfcco); 
% [W2, pc2] = pca(plps); 







feature = [mfcco(:,1:7), mfcco(:,14:19),mfcco(:,27:32),plps(:,1:8),plps(:,14:19),plps(:,27:32)]; % 164,78
% 
% 
% [W, pc] = pca(feature); 
% 
% feature = pc(:,1:39);



%feature =[pc1(:,1:7),pc1(:,14:19),pc1(:,27:32),pc2(:,1:8),pc2(:,14:19),pc2(:,27:32)];








end



% END OF PROGRAM
% ---------------------------------------------------------------
% --------------------------------------------------------------- % START FUNCTION DEFINITIONS
function mels = mel(freq)
% change frequency from Hz to mel 
mels = 1127 * log( 1 + (freq/700) ); 
end
% ---------------------------------------------------------------
function wins = sig2fm(input, winlen, winshft, frameno)
% put vector into matrix, each column is a frame.
% The rest of signal that is less than one frame is discarded
% winlen, winshft are in number of sample, notice winshft is not limited to % integer
input = input(:);
wins=zeros(winlen, frameno);
    for i = 1 : frameno
        b = round((i-1) * winshft);
        c = min(winlen, length(input) - b);
        wins(1:c,i) = input(b+1 : min(length(input), b+winlen));
    end
end
% ---------------------------------------------------------------
function fbank = initfiltb(framelen,numchan,fsamp,nfft) % triangle shape melfilter initialization
fftfreqs = ((0:(nfft/2-1))/nfft)*fsamp; % frequency of each fft point (1- fsamp/2)
melfft = mel(fftfreqs); % mel of each fft point
mel0 = 0;
mel1 = mel(fsamp/2); % highest mel
melmid = ((1:numchan)/(numchan+1))*(mel1-mel0) + mel0; % middle mel of each filter
fbank = zeros(numchan,nfft/2);
% non overlaping triangle window is used to form the mel filter
    for k = 2:(nfft/2) % for each fft point, to all the filters,do this:
        chan = max([ 0 find(melfft(k)>melmid) ]); % the highest index of melfft that is larger than the middle mel of all channels
        if(chan==0) % only the first filter cover here 
            fbank(1,k) = (melfft(k)-mel0)/(melmid(1)-mel0);
        elseif(chan==numchan) % only the last filter covered here 
            fbank(numchan,k) = (mel1-melfft(k))/(mel1-melmid(chan));
        else % for any other part, there will be two filter cover that frequency, in the complementary manner
        fbank(chan,k) = (melmid(chan+1)-melfft(k))/(melmid(chan+1)-melmid(chan));
        fbank(chan+1,k) = 1-fbank(chan,k); % complementary end
        end
    end
end
% ---------------------------------------------------------------
function dt = deltacc(input, winlen)
% calculates derivatives of a matrix, whose columns are feature vectors
tmp = 0;
for cnt = 1 : winlen
    tmp = tmp + cnt*cnt;
end
nrm = 1 / (2*tmp);
dt = zeros(size(input));
rows = size(input,1); 
cols = size(input,2);
for col = 1 : cols
    for cnt = 1 : winlen
        inx1 = col - cnt; inx2 = col + cnt;
        if inx1 < 1; inx1 = 1; end
        if inx2 > cols; inx2 = cols; end
        dt(:, col) = dt(:, col) + (input(:, inx2) - input(:, inx1)) * cnt;
    end
end
dt = dt* nrm;
end

% END FUNCTION DEFINITIONS
% ---------------------------------------------------------------