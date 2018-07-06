function [F0, score] = fast_mbsc_fixedWinlen_tracking(y,fs,do_tracking,vThres,wintime)
% Multi-band summary correlogram (MBSC) pitch detector for speech.
% This algorithm is described in the paper:
% L. N. Tan, A. Alwan, "Multi-band summary correlogram-based pitch detection 
% for noisy speech", Speech Communication, in press.
% Please cite the above reference, and this MBSC code package (with the 
% SPAPL shareware URL) in your work if you make use this code.
%
% For this code, a fixed window length specified by "wintime" is used for 
% all comb channel processing, and frame-by-frame mean-normalization is 
% not performed to reduce computational time. Your need to set the threshold
% and wintime that are appropriate for your task. wintime should be at least 
% twice the maximum period.
% If do_tracking is set to 0, a constant threshold is applied to the max.
% peak amplitude of MBSC to perform voicing detection, and the estimated
% F0 value is computed from the lag position of this peak.
% If do_tracking is set to 1, an SNR adaptive threshold with max threshold 
% of "vThres" is used to identify initial voiced segments. After which,
% pitch continuity tracking is performed on a per-voiced segment basis, 
% beginning at the frame with highest score in the segment. The most
% likely pitch estimate of this frame is used for continuity tracking. The
% initial voicing boundaries can expand or contract depending on the pitch
% continuity among the F0 candidates across time.
%
% Inputs: y - input samples
%        fs - sampling rate in Hz
%        do_tracking - set to 1 to perform pitch continuity tracking,
%                      set to 0 to perform constant threshold voicing
%                      detection.  Default = 1;
%        vthres - threshold for voiced/unvoiced detection. Default = 0.45
%        wintime - window length in seconds. Default = 0.06
%
% Outputs: F0 - Estimated F0 value, frame shift = 10ms. 
%               If F0=0, it denotes an unvoiced frame. 
%          score - peak amplitude of the MBSC which denotes the frame's
%                  degree of voicing.
%
% Written by Lee Ngee Tan
% University of California (Los Angeles)
% Date: Apr 25, 2013
% Email: ngee@seas.ucla.edu
%==========================================================================
if nargin<3
    do_tracking = 1;
    vThres = 0.45;
    %wintime = 0.06; 
    wintime = 0.03; 
elseif nargin<4
    vThres = 0.45;
    wintime = 0.06;
elseif nargin<5
    wintime = 0.06;
end

% Resample data to 8 kHz if fs is not 8 kHz
if(fs~=8000)
    ytmp = resample(y,8000,fs);
    y = ytmp;
    fs = 8000;
    clear ytmp;
end

%timeshift = 0.02; %0.01
timeshift = 0.01;
winshift = timeshift*fs;
minF0 = 50;
maxF0 = 200;  %400
if (maxF0>500)
    disp(['Maximum F0 should not exceed 500 Hz because multiple harmonics',...
        'will not be captured by the 1 kHz subband bandwidth']);
    return;
end
% A maximum of 500 Hz is allowed for the current design.
minlag = floor(fs/maxF0);
maxlag = ceil(fs/minF0);
n_sb = 4;

y_len = length(y);
n_fr = floor(y_len/winshift);
F0cand = zeros(n_fr,10);
cand_scores = zeros(n_fr,10);  
F0 = zeros(n_fr,1);
score = zeros(n_fr,1);

if do_tracking
    buf_len = 50;
    snr_buf = zeros(buf_len,1);
    sil_buf = zeros(buf_len,1);
    rel_buf = zeros(buf_len,1);
    snr_ctr = 0;
    sil_ctr = 0;
    rel_ctr = 0;
    snr_full = 0;
    sil_full = 0;
    rel_full = 0;
    Eratio_buf = zeros(buf_len,1);
    Eratio_ctr = 0;
    Eratio_full = 0;
    mean_snr = 20;
    mean_Eratio = 0.25;
    v_startfr = 1; 
    vuvThres = vThres;
    upper_snrThres = 15;
    lower_snrThres = 7.5;
    init_vuv = zeros(n_fr,1);
    vuv = zeros(n_fr,1);
    seg2clear = 0;
end

lag = minlag-1:maxlag+1;
doublag = (maxlag+2):2:2*maxlag;
lagRange = [lag, doublag];
n_ch = length(lagRange);
winlen = round(wintime*fs);
nfft = 4096;
n_Rlag = min(winlen-1,maxlag);
lagwt_line = polyfit([1,maxlag],[1,0.3],1);
n_stream = n_sb+1;
sb_ch_snr_fr = zeros(n_ch,n_fr,n_stream);
smooth_sb_ch_snr = zeros(n_ch,n_stream);
minSNRpk = 1.0;
sR_smooth = zeros(1,maxlag+1);

sb_y = zeros(y_len,n_stream);
halfFs = fs/2;
n_firpts = 32;
sb_fir = zeros(n_sb,n_firpts+1);
sb_fir(1,:) = fir1(n_firpts, 1000/halfFs);
sb_fir(2,:) = fir1(n_firpts,[800, 1800]/halfFs);
sb_fir(3,:) = fir1(n_firpts,[1600, 2600]/halfFs);
sb_fir(4,:) = fir1(n_firpts,[2400, 3400]/halfFs);

[combFiltmat, noise_combFiltmat, n_posfft]...
                             = gen_LPcombFBKs(lagRange, nfft, fs);

sb_specgram = zeros(n_posfft,n_fr,n_stream);
ext_zeros = zeros((winlen/winshift/2-0.5)*winshift,n_stream);
hamwin = hamming(winlen)*ones(1, n_fr);                         
                         
for sb = 1:n_sb
    tmp = filter(sb_fir(sb,:),1,[y;zeros(n_firpts/2,1)]);
    if(sb==1)
        sb_y(:,sb) = tmp(n_firpts/2+1:end);
    end

    tmp3 = hilbert(tmp(n_firpts/2+1:end));
    tmp3 = (real(tmp3)).^2+(imag(tmp3)).^2;
    sb_y(:,sb+1) = tmp3;
end
ext0_sb_y = [ext_zeros;sb_y;ext_zeros];

for sb = 1:n_stream

    frmwin = sig2fm(ext0_sb_y(:,sb), winlen, winshift, n_fr);      
    % Hamming window each frame
    frmwin = frmwin .* hamwin;
    % FFT
    ffto = fft(frmwin, nfft);
    tmp2 = ffto(1:n_posfft,:);
    tmp2 = (real(tmp2)).^2+(imag(tmp2)).^2;
    sb_specgram(:,:,sb) = tmp2;
    sb_ch_sigpow = combFiltmat'*tmp2;
    sb_ch_noisepow = noise_combFiltmat'*tmp2;
    sb_ch_snr_fr(:,:,sb) = (sb_ch_sigpow./sb_ch_noisepow);
end

smooth_snrWt_normR = zeros(n_stream,maxlag+1);
lim2 = find(lagRange==maxlag+1);
    
for fr = 1:n_fr
    smooth_sb_snr_wt = zeros(n_ch,n_stream);
    smooth_max_chan_snr = zeros(1,n_stream);

    for sb = 1:n_stream

       fr_posf_spec = sb_specgram(:,fr,sb);

        if(fr==1) % 1st update
            smooth_sb_ch_snr(:,sb) = sb_ch_snr_fr(:,fr,sb);
        else
            smooth_sb_ch_snr(:,sb) = 0.5*(sb_ch_snr_fr(:,fr,sb) + smooth_sb_ch_snr(:,sb));
        end

    
    % Perform channel selection
        [snrpk,snrpkloc] = findpeaks_fast(smooth_sb_ch_snr(:,sb),minSNRpk);
        
        if(~isempty(find(snrpkloc<lim2, 1)))
            snrpklag = lagRange(snrpkloc);
            n_pks = length(snrpklag);
            validPks = false(n_pks,1);
            sum_validPks = 0;
            n_pks = sum(snrpklag<=maxlag);
            for pk = 1:n_pks
                lag_dev = abs(0.5*snrpklag/snrpklag(pk)-1);
                doublagloc = find(lag_dev<0.2);
                if(~isempty(doublagloc))
                    if(length(doublagloc)>1)
                        [min_val,min_idx] = min(lag_dev(doublagloc));
                        validPks([pk,doublagloc(min_idx)])=1;
                        sum_validPks = 1;
                    else
                        validPks([pk,doublagloc])=1;
                        sum_validPks = 1;
                    end
                end
            end
            
            if(sum_validPks)
                valid_snrpklocs = snrpkloc(validPks);
                smooth_sb_snr_wt(valid_snrpklocs,sb) = snrpk(validPks)-1;
            else
                smooth_snrWt_normR(sb,:) = 0;
                smooth_max_chan_snr(sb) = 0;
                continue
            end
        else
            smooth_snrWt_normR(sb,:) = 0;
            smooth_max_chan_snr(sb) = 0;
            continue
        end    
        
        % Compute energy-normalized autocorrelation for selected channels
        pkChIdx = find(smooth_sb_snr_wt(:,sb)>0);
        select_wt = smooth_sb_snr_wt(pkChIdx,sb);
        n_pkChIdx = length(pkChIdx);
        combfilt_powspec = (fr_posf_spec*ones(1,n_pkChIdx)).*combFiltmat(:,pkChIdx);
        tmp = ifft(combfilt_powspec,nfft,'symmetric');
        tmp_sb_normR = tmp(2:n_Rlag+2,:)./(ones(n_Rlag+1,1)*tmp(1,:));
        [maxpk_allCh,pklag_allCh] = findMaxPk_inMatCols(tmp_sb_normR);
        for pkch = 1:n_pkChIdx
            ch = pkChIdx(pkch);
            pklag = pklag_allCh(pkch);
            if(pklag~=0)
                tol = 0.2*lagRange(ch);
                if(abs(lagRange(ch)-pklag)>tol && abs(lagRange(ch)-2*pklag)>tol)
                    select_wt(pkch) = 0;
                end
            else
                select_wt(pkch) = 0;
            end
        end
        % Compute subband summary correlogram 
        sum_select_wt = sum(select_wt);
        if(sum_select_wt>0)
            maxpk = max(select_wt);
            smooth_max_chan_snr(sb) = maxpk;
            select_wt = select_wt/sum_select_wt;
            smooth_snrWt_normR(sb,:) = (tmp_sb_normR*select_wt)';
        else
            smooth_snrWt_normR(sb,:) = 0; 
        end        
    end
        
    % Calculate between-subband reliability
    smooth_ACmaxpk_lag = zeros(1,n_stream);
    smooth_sb_rel_ctr = ones(1,n_stream);
    [maxpk_allSb,pkIdx_allSb] = findMaxPk_inMatCols(smooth_snrWt_normR');    
    for sb = 1:n_stream
        pkIdx = pkIdx_allSb(sb);
        if(pkIdx~=0)
            smooth_ACmaxpk_lag(sb) = pkIdx;
        end
    end
    for sb = 1:n_stream
        lag_ratio = smooth_ACmaxpk_lag/smooth_ACmaxpk_lag(sb);
        smooth_sb_rel_ctr(sb) = sum(abs(1-lag_ratio)<0.1);
    end
    
    % Compute multi-band summary correlogram 
    if(sum(smooth_max_chan_snr))
        smooth_max_chan_snr = smooth_max_chan_snr.*smooth_sb_rel_ctr;
        smooth_max_chansnr_wt = smooth_max_chan_snr/sum(smooth_max_chan_snr);
        sR_smooth_sb_snr = smooth_max_chansnr_wt*smooth_snrWt_normR;
    else
        sR_smooth_sb_snr = zeros(1,maxlag+1);
    end
    
    if(sR_smooth(1)==0) % 1st update of sR_smooth 
        sR_smooth = sR_smooth_sb_snr;
    else
        sR_smooth = 0.5*(sR_smooth + sR_smooth_sb_snr);
    end
    
    % Peak extraction and lag-weighting
    [pks, pklocs] = findpeaks_fast(sR_smooth(minlag-1:maxlag+1),0,'descend');
    if(~isempty(pks))
        pklocs = pklocs + minlag - 2;
        len = length(pks);
        n_cands = min(len,10);
        lag_cands = zeros(n_cands,1);
        curr_F0cand = zeros(n_cands,1);
        curr_scores = zeros(n_cands,1);
        for m = 1:n_cands
            fit_x = pklocs(m)+[-1,0,1];
            p_coeff = polyfit(fit_x,sR_smooth(fit_x),2);
            x_turnPt = -0.5*p_coeff(2)/p_coeff(1);
            y_turnPt = polyval(p_coeff,x_turnPt); 
            lag_cands(m) = x_turnPt;
            curr_F0cand(m) = fs./x_turnPt;                
            curr_scores(m) = y_turnPt;
        end
        
        F0cand(fr,1:n_cands) = curr_F0cand';
        cand_scores(fr,1:n_cands) = curr_scores';
        score(fr) = curr_scores(1);
    end
    
    if do_tracking
        fr_start = (fr-1)*winshift+1;
        fr_end = fr*winshift;
        y_Efr = y(fr_start:fr_end);
        sb1_y_fr = sb_y(fr_start:fr_end,1);
        % Calculate frame energy
        E = sum(y_Efr.^2);
        % Calculate sb1 frame energy
        sb1_E = sum(sb1_y_fr.^2);
        sbRel= smooth_sb_rel_ctr(1);

        init_vuv(fr) = cand_scores(fr,1)>=vuvThres;
        if(fr<=5)  % treat init 5 frames as silence      
            sil_ctr = mod(sil_ctr,buf_len)+1;
            sil_buf(sil_ctr) = E;
            mean_sil_E = mean(sil_buf(1:sil_ctr));

            if(fr>2)
                vuv(fr-2) = median([zeros(5-fr,1);init_vuv(1:fr)]);
            end
        else
            if(init_vuv(fr)==0) % if current frame is detected as unvoiced
                if(E<=2*mean_sil_E)
                    sil_ctr = mod(sil_ctr,buf_len)+1;
                    sil_buf(sil_ctr) = E;
                    if(sil_ctr==buf_len)
                        sil_full = 1;
                    end
                    if(~sil_full)
                        mean_sil_E = mean(sil_buf(1:sil_ctr));
                    else
                        mean_sil_E = mean(sil_buf);
                    end

                    Eratio_ctr = mod(Eratio_ctr,buf_len)+1;
                    Eratio_buf(Eratio_ctr) = sb1_E/E;
                    if(Eratio_ctr==buf_len)
                        Eratio_full = 1;
                    end
                    if(~Eratio_full)
                        mean_Eratio = mean(Eratio_buf(1:Eratio_ctr));
                    else
                        mean_Eratio = mean(Eratio_buf);
                    end
                end
            else % if current frame is detected as voiced
                snr_ctr = mod(snr_ctr,buf_len)+1;
                snr_buf(snr_ctr) = 10*log10(E/mean_sil_E);
                if(snr_ctr==buf_len)
                    snr_full = 1;
                end
                if(~snr_full)
                    mean_snr = mean(snr_buf(1:snr_ctr));
                else
                    mean_snr = mean(snr_buf);
                end

                rel_ctr = mod(rel_ctr,buf_len)+1;
                rel_buf(rel_ctr) = sbRel;
                if(rel_ctr==buf_len)
                    rel_full = 1;
                end
                if(~rel_full)
                    median_rel = median(rel_buf(1:rel_ctr));
                else
                    median_rel = median(rel_buf);
                end

                if(mean_snr>upper_snrThres)  
                    vuvThres = vThres;
                elseif(mean_snr<lower_snrThres) 
                    if(mean_Eratio<0.5 && median_rel==1)
                        vuvThres = vThres-0.15;
                    elseif(mean_Eratio<0.5)
                        vuvThres = vThres-0.1;
                    else
                        vuvThres = vThres;
                    end
                else
                    vuvThres = vThres;
                end 

            end

            vuv(fr-2) = median(init_vuv(fr-4:fr));

            if(vuv(fr-3)==1 && vuv(fr-2)==0) % end of voiced segment
                v_endfr = fr-3;
                v_seg = v_startfr:v_endfr;
                if(mean_snr>=lower_snrThres)
                    F0(v_seg) = F0cand(v_seg,1);
                    score(v_seg) = cand_scores(v_seg,1);
                else
                    [min_score, max_add_fr,seg_pklocs] = deter_minScore_addFr(cand_scores(v_seg,1),median_rel,vThres);
                    if(max_add_fr)
                        first_fr = max(1,v_startfr-max_add_fr);
                        last_fr = min(n_fr,v_endfr+max_add_fr);   
                        v_seg = first_fr:last_fr;
                        seg_pklocs = seg_pklocs+v_startfr-first_fr;
                    end
                    if(fr>=v_seg(end))
                        [segF0, segScore] = F0_continuity(min_score,seg_pklocs,F0(v_seg),F0cand(v_seg,:),cand_scores(v_seg,:));
                        F0(v_seg) = segF0;
                        score(v_seg) = segScore;
                    else
                        seg2clear = 1;
                    end
                end
            elseif(vuv(fr-3)==0 && vuv(fr-2)==1) % start of voiced segment
                v_startfr = fr-2;
            end

            if(seg2clear && fr==v_seg(end))
                seg2clear = 0;
                [segF0, segScore] = F0_continuity(min_score,seg_pklocs,F0(v_seg),F0cand(v_seg,:),cand_scores(v_seg,:));
                F0(v_seg) = segF0;
                score(v_seg) = segScore;
            end


            if(fr>=n_fr-1)
                vuv(fr) = median([init_vuv(fr-2:n_fr);zeros(2+fr-n_fr,1)]);
                if(vuv(fr-1)==0 && vuv(fr)==1) % start of voiced segment
                    v_startfr = fr;
                elseif(vuv(fr-1)==1 && vuv(fr)==0) % end of voiced segment
                    v_endfr = fr-1;
                    v_seg = v_startfr:v_endfr;
                    if(mean_snr>=lower_snrThres)
                        F0(v_seg) = F0cand(v_seg,1);
                        score(v_seg) = cand_scores(v_seg,1);
                    else
                        [min_score, max_add_fr,seg_pklocs] = deter_minScore_addFr(cand_scores(v_seg,:),median_rel,vThres);
                        if(max_add_fr)
                            first_fr = max(1,v_startfr-max_add_fr);
                            last_fr = min(n_fr,v_endfr+max_add_fr);    
                            v_seg = first_fr:last_fr;
                            seg_pklocs = seg_pklocs+v_startfr-first_fr;
                        end
                        if(fr>=v_seg(end))
                            [segF0, segScore] = F0_continuity(min_score,seg_pklocs,F0(v_seg),F0cand(v_seg,:),cand_scores(v_seg,:));
                            F0(v_seg) = segF0;
                            score(v_seg) = segScore;
                        else
                            seg2clear = 1;
                        end    
                    end
                end
            end      
        end
    end
end

if do_tracking==0
    % Perform V/UV detection and F0 candidate selection
    vuv = cand_scores(:,1)>=vThres;
    vuv = medfilt1(vuv,5);
    for fr = 1:n_fr
        cands = cand_scores(fr,:)>0;
        if(sum(cands))
            lags = (fs./F0cand(fr,cands));
            lagwt = polyval(lagwt_line,lags);
            wt_scores = cand_scores(fr,cands).*lagwt;
            [max_wt_score, max_idx] = max(wt_scores);
            F0(fr) = F0cand(fr,max_idx(1));
            score(fr) = cand_scores(fr,max_idx(1));
        end
    end
    F0 = F0.*vuv;

    % Perform F0-tracking
elseif v_startfr>v_endfr
    v_endfr = n_fr;
    v_seg = v_startfr:n_fr;
    if(mean_snr>=lower_snrThres)
        F0(v_seg) = F0cand(v_seg,1);
        score(v_seg) = cand_scores(v_seg,1);
    else
        [min_score, max_add_fr,seg_pklocs] = deter_minScore_addFr(cand_scores(v_seg,:),median_rel,vThres);
        if(max_add_fr)
            first_fr = max(1,v_startfr-max_add_fr);
            last_fr = min(n_fr,v_endfr+max_add_fr);    
            v_seg = first_fr:last_fr;
            seg_pklocs = seg_pklocs+v_startfr-first_fr;
        end
        [segF0, segScore] = F0_continuity(min_score,seg_pklocs,F0(v_seg),F0cand(v_seg,:),cand_scores(v_seg,:));
        F0(v_seg) = segF0;
        score(v_seg) = segScore;
    end
end

    
        
% Generate signal and noise-capturing comb filterbanks (FBK)
function [combFiltmat,noise_combFiltmat, n_posfft] = gen_LPcombFBKs(all_lags, nfft, Fs)

lp_freq = 1000;
fq = Fs./all_lags;
n_fq = length(fq); % Total # channels in each FBK
halfFs = Fs/2;
freq = (-halfFs+Fs/nfft:Fs/nfft:halfFs)';
twoPi = 2*pi;
freq0 = freq(nfft/2:end);
lowfcut_startIdxH = find(freq0>lp_freq, 1);
freq0 = twoPi*freq0;
n_posfft = lowfcut_startIdxH;
combFiltmat = zeros(n_posfft,n_fq); % comb_low
noise_combFiltmat = zeros(n_posfft,n_fq); % noise_comb_low
%expdecaycell = zeros(n_posfft,n_fq); % harmonic decreasing function
for ch = 1:n_fq      
    f_ratio = freq0/fq(ch);
    filt = 0.5*(1+cos(f_ratio));
    [lowfcutL,lowfcut_idxL] = min(filt(f_ratio>=0 & f_ratio<twoPi)); 
    trun_filt = filt(lowfcut_idxL:lowfcut_startIdxH);
    combFiltmat((lowfcut_idxL:lowfcut_startIdxH),ch) = trun_filt.^2;
    noise_combFiltmat((lowfcut_idxL:lowfcut_startIdxH),ch) = (1-trun_filt).^2;
    %expdecaycell(:,ch) = 1+[1;min(1,(fq(ch)./freq0(2:n_posfft)))];
    %lowfcut(ch) = lowfcut_idxL;
end

function wins = sig2fm(input,winlen,winshft,numwins)
% put vector into matrix, each column is a frame. 
% The rest of signal that is less than one frame is discarded
input = input(:);     
wins=zeros(winlen,numwins);
for i=0:numwins-1
  wins(:,i+1) = input(1+i*winshft:i*winshft+winlen);
end

function [pkMag, pkIdx] = findMaxPk_inMatCols(x)
[n_row, n_col] = size(x);
x1 = x(1:n_row-2,:);
x2 = x(2:n_row-1,:);
x3 = x(3:n_row,:);
xL = x2>x1;
xR = x2>x3;
xLnR = xL&xR;
rowIdx = 2:n_row+1;
pkMag = zeros(n_col,1);
pkIdx = zeros(n_col,1);
for n = 1:n_col
    pklocs = rowIdx(xLnR(:,n));   
    if(~isempty(pklocs))
         pks = x(pklocs,n);
        [pkMag(n), idx] = max(pks);
        pkIdx(n) = pklocs(idx);
    end
end

function [pks, pklocs] = findpeaks_fast(y,minPkMag,sortstr)
x = y(:);
x1 = x(1:end-2);
x2 = x(2:end-1);
x3 = x(3:end);
xL = x2>x1;
xR = x2>x3;
rowIdx = 2:length(y);
pklocs = rowIdx(xL&xR);
%pklocs = find(xL&xR)+1;
pks = x(pklocs);
retain_pkIdx = pks>=minPkMag;
if(~isempty(retain_pkIdx))    
    pks = pks(retain_pkIdx);
    pklocs = pklocs(retain_pkIdx);
    if(nargin>2)
        if(strcmp(sortstr,'descend'))
            [sortPks,sortIdx] = sort(pks,'descend');   
            pks = sortPks;
            pklocs = pklocs(sortIdx); 
        elseif(strcmp(sortstr,'ascend'))
            [sortPks,sortIdx] = sort(pks);
            pks = sortPks;
            pklocs = pklocs(sortIdx); 
        end
    end
else
    pks = [];
    pklocs = [];
end

function [min_score, max_add_fr, pklocs] = deter_minScore_addFr(scores,median_rel,vThres)
v_len = size(scores,1);
if(v_len<3)
    [pks, pklocs] = max(scores(:,1));
else       
    %[pks, pklocs] = findpeaks(scores(:,1),'sortstr','descend');
    [pks,pklocs] = findpeaks_fast(scores(:,1),-1,'descend');
    if(isempty(pklocs))
        [pks, pklocs] = max(scores(:,1));
    end
end
if(median_rel==1)
    min_score = vThres-0.25;
    max_add_fr = 4;
else
    min_score = vThres-0.2;
    max_add_fr = 2;
end

function [F0, F0score] = F0_continuity(min_score,pklocs,F0,F0_cand,scores)
F0_dev_tol = 0.1;
n_fr = length(F0);
F0score = scores(:,1);
curr_pk_idx = 0;
while(curr_pk_idx<length(pklocs))
    curr_pk_idx = curr_pk_idx + 1;
    if(F0(pklocs(curr_pk_idx))==0)
        F0(pklocs(curr_pk_idx)) = F0_cand(pklocs(curr_pk_idx),1);
        F0score(pklocs(curr_pk_idx)) = scores(pklocs(curr_pk_idx),1);
    end

    if(pklocs(curr_pk_idx)+1<=n_fr)
        if(F0(pklocs(curr_pk_idx)+1)==0)
            % get F0 for frames to the right of pk
            for n = pklocs(curr_pk_idx)+1:n_fr
                if(F0(n)~=0)
                    break;
                end
                curr_scores = scores(n,scores(n,:)>min_score);
                n_cand = length(curr_scores);
                curr_F0_cand = F0_cand(n,1:n_cand);
                prev_F0 = F0(n-1);
                [minF0diff,minIdx] = min(abs(curr_F0_cand - prev_F0));
                if(minF0diff/prev_F0<=F0_dev_tol)
                    F0(n) = F0_cand(n,minIdx);
                    F0score(n) = curr_scores(minIdx);
                else
                    if(n-2>0)
                        prev_F0_change = prev_F0-F0(n-2);
                        curr_F0_change = F0_cand(n,minIdx)-prev_F0;
                        changeDiff = abs(curr_F0_change-prev_F0_change)/prev_F0;
                        if(changeDiff<=F0_dev_tol)
                            F0(n) = F0_cand(n,minIdx);
                            F0score(n) = curr_scores(minIdx);
                        else
                            break;
                        end
                    end
                end

            end
        end
    end
    
    if(pklocs(curr_pk_idx)-1>=1)
        if(F0(pklocs(curr_pk_idx)-1)==0)
            % get F0 for frames to the left of pk
            for n = pklocs(curr_pk_idx)-1:-1:1
                if(F0(n)~=0)
                    break;
                end

                curr_scores = scores(n,scores(n,:)>min_score);
                n_cand = length(curr_scores);
                curr_F0_cand = F0_cand(n,1:n_cand);
                prev_F0 = F0(n+1);
                [minF0diff,minIdx] = min(abs(curr_F0_cand - prev_F0));
                if(minF0diff/prev_F0<=F0_dev_tol)
                    F0(n) = F0_cand(n,minIdx);
                    F0score(n) = curr_scores(minIdx);
                else
                    if(n+2<=n_fr)
                        prev_F0_change = prev_F0-F0(n+2);
                        curr_F0_change = F0_cand(n,minIdx)-prev_F0;
                        changeDiff = abs(curr_F0_change-prev_F0_change)/prev_F0;
                        if(changeDiff<=F0_dev_tol)
                            F0(n) = F0_cand(n,minIdx);
                            F0score(n) = curr_scores(minIdx);
                        else
                            break;
                        end
                    end
                end
            end
        end
    end
    
end