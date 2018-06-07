%rawdata=template;
%Fs=8000;
function lpccs = LPCC(rawdata,Fs)
    winlen = 25;
    winshft =10;
    cepnum  =13;
    liftercoe =22;
    numchan =26;
    preemcoeff =0.97;
    deltawindow =2;
    accwindow =2;
    C0 =0;
    usedeltas =1; 
    useenergy =0;
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
nfft = 2*(2^nextpow2(winlen)); % FFT size % initialize MEL filter bank
fbank = initfiltb(winlen, numchan, fsamp, nfft);
% initialize lifter coefficients
lifter = (1 + (liftercoe/2)*sin((pi/liftercoe)*(0:cepnum)) );
% pre-emphasis
am = [1 0]; % denominator polynomial
bm = [1 -preemcoeff]; % numerator polynomial 
preem = filter(bm, am, input);
% change signal (a vector) into frame (a matrix), where each collum is a frame
frmwin = sig2fm(preem, winlen, winshft, FrameNo);
[winlen, framenum] = size(frmwin);
% Hamming window each frame
frmwin = frmwin .* (hamming(winlen) * ones(1, framenum));
% Log energy
LE = log(sum(frmwin.*frmwin));
% FFT
fft_mag = zeros(cepnum, framenum); 
for i = 1:framenum
    lpcc = msf_lpcc(frmwin(:, 1),8000,'order',13);
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