%rawdata=template;
%Fs=8000;
function feature = CONCAT(rawdata,Fs)

feature1 = PLP(rawdata,Fs);
feature2 = PNCC3(rawdata,Fs);
feature = [feature1(:,1:9), feature1(:,14:20),feature1(:,27:33),feature2(:,1:6), feature2(:,14:18),feature1(:,27:31)];
    
end

% END FUNCTION DEFINITIONS
% ---------------------------------------------------------------