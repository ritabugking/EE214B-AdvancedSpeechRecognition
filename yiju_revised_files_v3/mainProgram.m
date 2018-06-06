%% Main Program

if ~exist('features/train', 'dir')
    mkdir features/train
end

if ~exist('features/test', 'dir')
    mkdir features/test
end

if ~exist('models', 'dir')
    mkdir models
end

%% Training
disp('####################  TRAINING   ###################');
tic
train_clean
toc


%% Testing
disp('####################  TESTING   ###################');
tic
test_male_clean
test_female_clean
test_male_10dB
test_female_10dB
test_male_5dB
test_female_5dB
toc
