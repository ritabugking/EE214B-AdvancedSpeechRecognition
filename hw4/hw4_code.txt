% clearvars;
% close all;
nntraintool('close');

v = ver;
if ~any(strcmp(cellstr(char(v.Name)), 'Neural Network Toolbox'))
    disp('Please install Neural Network Toolbox from Add-Ons')
    return;
end

files = dir('recordings/*.wav');
max_samples = 0;
num_classes = 10;

% Find the audio file with maximum samples
for i = 1:length(files)
    info = audioinfo([files(i).folder  '/' files(i).name]);
    if info.TotalSamples > max_samples
        max_samples = info.TotalSamples;
    end
end

% Dimensions of data
data_dim = numel(MFCC(zeros(max_samples, 1), info.SampleRate));

% Initialize data and labels
data = zeros(data_dim, length(files));
labels = zeros(num_classes, length(files));

% Generate MFCC features
for i = 1:length(files)
    % Read file
    [snd, fs] = audioread([files(i).folder  '/' files(i).name]);
    
    % Stereo to mono
    if size(snd, 2) == 2
        snd = snd(:, 1)/2 + snd(:, 2)/2;
    end
    
    labels(1 + str2double(files(i).name(1)), i) = 1;
    
    % Zero pad the rest of the utterance 
    snd1 = zeros(max_samples, 1);
    snd1(1:length(snd)) = snd;
    
    % Calculate MFCC features for each frame and concatenate them
    mfcc = MFCC(snd1, fs);
    data(:, i) = mfcc(:);
end

% Fix the random seed
RandStream.setGlobalStream(RandStream ('mrg32k3a','Seed', 1234));

% Create a Neural Network Model
% Each integer represents number of hidden neurons 
% for a fully connected layer
% In this example, there are 2 hidden layers with 10 hidden neurons
net = patternnet([14]);

% Fix the random seed
RandStream.setGlobalStream(RandStream ('mrg32k3a','Seed', 1234));

% Set up Division of Data for Training, Validation, Testing
%train_ratio = 50/100;
train_ratio = 70/100;
%val_ratio = 20/100;
val_ratio = 0/100;
test_ratio = 30/100;
net.divideFcn = 'divideind';
[trainInd,valInd,testInd] = divideint(1500,train_ratio,val_ratio,test_ratio);
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% Regularization
%net.performParam.regularization = 0.001;
net.performParam.regularization = 0.4;

% Train the Network
[net,tr] = train(net,data,labels);

% Test the Network
outputs = net(data);
errors = gsubtract(labels,outputs);
performance = perform(net,labels,outputs)

[testacc, cm] = calcCM(labels, outputs, testInd)
% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
