 function feature = PLP(rawdata,Fs)

 data=rawdata;
[feature, spectra, pspectrum, lpcas, F, M] = rastaplp(data,Fs,0,12);
     % calculate 1st derivative (velocity)
dt1 = deltacc(feature, 2);

% calculate 2nd derivative (acceleration)
dt2 = deltacc(dt1, 2);
% append dt1 and dt2 to mfcco
feature = [feature; dt1; dt2];
feature = feature';



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