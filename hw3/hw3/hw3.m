% YI-JU WANG - 204617899

%% clear the workspace
%clear all; close all; clc;
label_mat=[];
digit_err=[];
for digit = 1:10
    err=0;
    for num_testfile = 1:49
        filename_test = sprintf('%d%s%d%s', digit-1, '_jackson_', num_testfile, '.wav');
        [test, fs] = audioread(filename_test);
        [mfcc_test, d1, d2] = MFCC(test,8000);
        %feature_test = [mfcc_test; d1; d2]; % MFCCs
        feature_test = LPCC(test,fs); % LPCCs
        min_dis=Inf;
        label=-1;
        for num_file=1:10
            filename_template = sprintf('%d%s', num_file-1, '_jackson_0.wav'); % input audio filename
            [template, fs] = audioread(filename_template);

            [mfcc_template, d1, d2] = MFCC(template,8000);
            %feature_template=[mfcc_template; d1; d2]; % MFCCs
            feature_template=LPCC(template,fs); % LPCCs
            [dist,ix,iy] = dtw(feature_template, feature_test); % Euclidean distance
            %[dist,ix,iy] = dtw(feature_template, feature_test, 'absolute'); % Manhattan distance
            if dist < min_dis
                min_dis=dist;
                label = num_file -1;
            end
        end
        label_mat=[label_mat; label];
        if label~=digit-1
            err = err + 1;
        end
    end
    digit_err=[digit_err; err];
end

t=[0:9];
t=repmat(t,49,1);
t=t(:);
labels=[label_mat, t];
acc = sum(labels(:,1)==labels(:,2))/size(labels,1);
err = 1-sum(labels(:,1)==labels(:,2))/size(labels,1);
disp(acc)

