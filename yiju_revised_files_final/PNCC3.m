function feature = PNCC3(rawdata,Fs)

winlen      = 25;            % window length in 100 nsec
if nargin == 3
    winshft = 1000/120;      
else
    winshft = 10;            % window shift in 100 nsec %samples
end
nfft        = 512;            % fft size 
cepnum      = 13;             % number of cepstral coefficients
liftercoe   = 22;             % liftering coefficient
numchan     = 40;             % number of channels of the MEL filter bank 26
preemcoeff  = 0.97;           % coefficient for pre-emphasis
deltawindow = 2;              % window length to calculate 1st derivatives
accwindow   = 2;              % window length to calculate 2nd derivatives
%C0          = 1;              % use zeroth cepstral coefficient (0/1)

% PNCC parameters ---------------------------------------------

power_coeff = 1/20;
excite_constant = 2;

% asymmetric noise suppression constants. 
lambda_a = 0.999;
lambda_b = 0.5;

% temporal masking constants
masking_forget_factor = 0.85;
decay_factor = 0.15;

% spectral smoothing parameters
ss_length = 3;

% power normalization parameters
power_forget_factor = 0.999;

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
%lifter = (1 + (liftercoe/2)*sin((pi/liftercoe)*(0:cepnum)) );

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



% % =========== gender identify ==========
% feature=MFCC(rawdata,Fs);  
%     
%     %BestModelMale=load('BestModelMale.mat');
%     TestMFCCs = feature;
%     % Calculate PDF for male and female GMMs
%     ProbsMale = pdf(BestModelMale, TestMFCCs);          
%     ProbsFemale = pdf(BestModelFemale, TestMFCCs);
%     ProbsAll = ProbsMale+ ProbsFemale;
%     prob_m = ProbsMale./ProbsAll;
%     prob_f = ProbsFemale./ProbsAll;
%     %averageMale = mean(ProbsMale);
%     ave_prob_m = mean(prob_m);
%     ave_prob_f = mean(prob_f);
%     %averageFemale = mean(ProbsFemale);
% %     m=[m;averageMale];
% %     f=[f;averageFemale];
% %     p_m =[p_m; ave_prob_m];
% %     p_f=[p_f; ave_prob_f];
% %     counterMale = 0;
% %     counterFemale = 0;
% %     
% %     for j = 1:length(ProbsMale)
% %         if (ProbsMale(j) > ProbsFemale(j))
% %             counterMale = counterMale + 1;
% %         else
% %             counterFemale = counterFemale + 1;
% %         end
% %     end
%     
%     %if (averageMale > averageFemale)
%     if (ave_prob_m > ave_prob_f)    
%         %classification_m{index} = 'M';
%         ;
%     else
%         %classification_m{index} = 'F';
%         % ===================== vtln ==================
%         alpha=1.153;
%         ffto2= ffto;
%         a=struct();
%         for i=1:size(ffto2,2)
%         a(i).data=ffto2(:,i);
%         end
%         aa=vtln(a, 'asymmetric', alpha);
%         ffto3=[];
%         for i=1:size(ffto2,2)
%             ffto3=[ffto3, aa(i).data];
%         end
% 
%         ffto = ffto3;
%     end

% =================================================



% %vtln
% alpha=1.153;
% ffto2= ffto;
% a=struct();
% for i=1:size(ffto2,2)
% a(i).data=ffto2(:,i);
% end
% aa=vtln(a, 'asymmetric', alpha);
% ffto3=[];
% for i=1:size(ffto2,2)
%     ffto3=[ffto3, aa(i).data];
% end
% 
% ffto = ffto3;



% MEL filtering (currently replacing the gammatone)
%fb = fbank*ffto(1 : (nfft/2), :);
p = fbank.^2 * ffto(1:(nfft/2),:).^2; % check if this is correct? 

mpmat = movavg(p, 2); % M = 2 according to paper. can change. 
q = p * mpmat;

% figure
% hold on
% for i=1:size(mp, 1)
%     %subplot(size(mp, 2), i, 1)
%     plot(mp(i,:));
% end

qle = anr_filter(q, lambda_a, lambda_b); % can change these values. 
                                    % 1 > b > a > 0 means upper envelope. 
                                    % 1 > a > b > 0 means lower envelope. 
                                    

q0 = q - qle;        % noise removal. 
q0(q0<0) = 0;           % half wave rectification. 

% following portion deals with zero intervals and small local variances.
qf = anr_filter(q0, lambda_a, lambda_b); 
qtm = temporal_mask(q0, masking_forget_factor, decay_factor); % temporal masking (can change vals)
rsp = max(qtm, qf);

excite = q >= excite_constant*qle; % check for excitation using noise. 
r = excite.*rsp + ~excite.*qf;
s = spectral_smoothing(q, r, ss_length);
t = p.*s;

mu = mean_power(t, power_forget_factor); % not sure why it's 0.999 but it's apparently data dependent.
% might need VAD as well. 
k = 1; % supposedly this is arbitrary. 
u = k * t ./ mu'; 
v = u .^ (power_coeff); % there's also an approximation. 

% do liftering with a lifted sin wave
%v = v .* (lifter' * ones(1, framenum));

pncco = dct(v);
pncco((cepnum+1):numchan, :) = [];

%=====================================================================
%================= End of Noise Robust Processing ====================
%=====================================================================

% calculate 1st derivative (velocity)
dt1 = deltacc(pncco, deltawindow);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(pncco, accwindow);
% append dt1 and dt2 to mfcco
pncco = [pncco; dt1; dt2];
feature = pncco';

% END OF PROGRAM
% ---------------------------------------------------------------
function fbank = initfiltb(framelen,numchan,fsamp,nfft)
% triangle shape melfilter initialization

fftfreqs = ((0:(nfft/2-1))/nfft)*fsamp;  % frequency of each fft point (1-fsamp/2)
melfft = mel(fftfreqs);   % mel of each fft point

mel0 = 0;                  
mel1 = mel(fsamp/2);       % highest mel 
melmid = ((1:numchan)/(numchan+1))*(mel1-mel0) + mel0; % middle mel of each filter

fbank = zeros(numchan,nfft/2);

% non overlaping triangle window is used to form the mel filter
for k = 2:(nfft/2)  % for each fft point, to all the filters,do this:
  chan = max([ 0 find(melfft(k)>melmid) ]); % the highest index of melfft that is larger than the middle mel of all channels
  if(chan==0)  % only the first filter cover here
    fbank(1,k) = (melfft(k)-mel0)/(melmid(1)-mel0);
  elseif(chan==numchan)  % only the last filter covered here
    fbank(numchan,k) = (mel1-melfft(k))/(mel1-melmid(chan));
  else                   % for any other part, there will be two filter cover that frequency, in the complementary manner
    fbank(chan,k) = (melmid(chan+1)-melfft(k))/(melmid(chan+1)-melmid(chan));
    fbank(chan+1,k) = 1-fbank(chan,k);  % complementary
  end
end

function mels = mel(freq)
% change frequency from Hz to mel
mels = 1127 * log( 1 + (freq/700) );

% ---------------------------------------------------------------
function wins = sig2fm(input, winlen, winshft, frameno)
% put vector into matrix, each column is a frame. 
% The rest of signal that is less than one frame is discarded
% winlen, winshft are in number of sample, notice winshft is not limited to
% integer
input = input(:);     
wins=zeros(winlen, frameno);

for i = 1 : frameno
    b = round((i-1) * winshft);
    c = min(winlen, length(input) - b);
    wins(1:c,i) = input(b+1 : min(length(input), b+winlen));
end


function dt = deltacc(input, winlen)
% calculates derivatives of a matrix, whose columns are feature vectors

tmp = 0;
for cnt = 1 : winlen
    tmp = tmp + cnt*cnt;
end
nrm = 1 / (2*tmp);

dt   = zeros(size(input));
rows = size(input,1);
cols = size(input,2);
for col = 1 : cols
    for cnt = 1 : winlen
        inx1 = col - cnt; inx2 = col + cnt;
        if inx1 < 1;     inx1 = 1;     end
        if inx2 > cols;  inx2 = cols;  end
        dt(:, col) = dt(:, col) + (input(:, inx2) - input(:, inx1)) * cnt;
    end
end
dt = dt * nrm;

