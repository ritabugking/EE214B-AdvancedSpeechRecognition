% clear all;
% close all;
% 
% % Trains GMM for each male and female training utterances from TIMIT
% % MFCCs are used as features
% % Classifies new utterances as either male or female based on GMMs
% 
% addpath('VOICEBOX');
% addpath('DATA');
% FileLength = 300;               % Number of files in test data
% NUM_MFCCs = 12;                 % number of MFCC coeffients to use
ave_acc_m=[];
ave_acc_f=[];
NUM_MIXTURES = 27;               % numbre of mixtures in GMMs
%10 fold, 15,    92.27     80.03
%5 fold, 15, f = 93.90 m = 81.55
%5 fold, 20,     95.24,    81.68
%5 fold, 25,     96,       81.33
%5 fold, 30,     95.14,    81.13
% ===============================
%5 fold, 25, 92.29, 92.48
%5 fold, 26, 90.48, 94.29
%5 fold, 27, 92.67, 93.62
%5 fold, 28, 90.76, 93.34
%5 fold, 30, 91.05, 94.39
% =======
% 88, 93.72
% 91.14, 93.82
Fs=8000;
Root='./';
num_fold =5;  % 5-fold cross validate
for k=1:num_fold 
kk=num_fold -k;
TestDataRoot=[Root 'database/male_/'];
TestFeatureRoot=[Root 'features/test/'];
testfiles=dir(TestDataRoot);
testfiles_m=testfiles(3:end);
features=dir(TestFeatureRoot);
for n=3:length(features)
    delete([TestFeatureRoot '/' features(n).name]);  
end;

MaleMFCCs = [];
numfiles_m_all=length(testfiles_m);
rand_m = randperm(numfiles_m_all);
numfiles_m=numfiles_m_all*0.7; % train set: test set =7:3
m_data_train=[];
% data_fold =ones(10,1);
% i=1;
% data_fold(i)=0;
%for num=1:numfiles
% 9-fold dataset
if (kk~=0)
for num=1:round(numfiles_m*kk/num_fold) % 4 fold training set
    %file_name=char(testfiles(num).name);
    file_name=char(testfiles_m(rand_m(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);
    
    % cepstral mean subtraction 
    feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;

%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;

    %feature=mean(MFCC(data,Fs));
    MaleMFCCs = [MaleMFCCs; feature];
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;
end
if (k~=1)
for num=round(numfiles_m*(kk+1)/num_fold):numfiles_m % 4 fold training set
    %file_name=char(testfiles(num).name);
    file_name=char(testfiles_m(rand_m(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);
    %feature=mean(MFCC(data,Fs));
    
        % cepstral mean subtraction 
%     feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;

%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;
    
    
    MaleMFCCs = [MaleMFCCs; feature];
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;
end

% % FID = fopen('CorrectLabels.txt');       % Read in correct labels for test data
% FID = fopen('TestData2Labels.txt');       % Read in correct labels for test data
% filenames = textscan(FID, '%s');
% fclose(FID);
% Labels = filenames{1};

    % Keeps track of correct and incorrect classifications
IncorrectCount = 0;
IncorrectMale = 0;
IncorrectFemale = 0;
CorrectMale = 0;
CorrectFemale = 0;

%         %% Create MALE GMM
% FID = fopen('TrainingMale.txt');
% filenames = textscan(FID, '%s');
% fclose(FID);
% files = filenames{1};
% MaleMFCCs = [];
% 
%     % Get MFCCs
% for i = 1:FileLength
%     
%     F = files{i};
%     [speech, fs] = audioread(F);
%     MFCCs = melcepst(speech, fs, 'Mtaz', NUM_MFCCs, 26);   
%     MaleMFCCs = [MaleMFCCs; MFCCs];
% end

    % Determine best fit GMM with AIC algorithm
% AIC = zeros(1, 5);
% GMModels = cell(1, 5);
% options = statset('MaxIter', 1000); 
% for k = 1:5
%     cInd = kmeans(MaleMFCCs, k, 'Options', options, 'EmptyAction', 'singleton');
%     GMModels{k} = fitgmdist(MaleMFCCs, k, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);
%     AIC(k)= GMModels{k}.AIC;
% end
% 
% [minAIC, numComponents] = min(AIC);
% BestModelMale = GMModels{numComponents};

    % Fit GMM model to MFCCs
options = statset('MaxIter', 1000);         % limit max itterations without convergence

    % use kNN to initalise and set covariance type to diagonal
cInd = kmeans(MaleMFCCs, NUM_MIXTURES, 'Options', options, 'EmptyAction', 'singleton');
BestModelMale = fitgmdist(MaleMFCCs, NUM_MIXTURES, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);

        %% create FEMALE GMM
TestDataRoot=[Root 'database/female/'];
TestFeatureRoot=[Root 'features/test/'];
testfiles=dir(TestDataRoot);
testfiles_f=testfiles(3:end);
features=dir(TestFeatureRoot);
for n=3:length(features)
    delete([TestFeatureRoot '/' features(n).name]);  
end;

FemaleMFCCs = [];
numfiles_f_all=length(testfiles_f);
rand_f = randperm(numfiles_f_all);
numfiles_f=numfiles_f_all*0.7;
f_data_train=[];
if (kk~=0)
for num=1:round(numfiles_f*kk/num_fold)
    
    file_name=char(testfiles_f(rand_f(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);
    %feature=mean(MFCC(data,Fs));
        % cepstral mean subtraction 
        
%     feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;

%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;
    
    FemaleMFCCs = [FemaleMFCCs; feature];
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;    
end
if (k~=1)
for num=round(numfiles_f*(kk+1)/num_fold):numfiles_f % 9 fold training set
    %file_name=char(testfiles(num).name);
    file_name=char(testfiles_f(rand_f(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);
        % cepstral mean subtraction 
        
%     feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;

%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;
    
    %feature=mean(MFCC(data,Fs));
    FemaleMFCCs = [FemaleMFCCs; feature];
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;
end              
% FID = fopen('TrainingFemale.txt');
% filenames = textscan(FID, '%s');
% fclose(FID);
% files = filenames{1};
% FemaleMFCCs = [];
% 
%     % Get MFCCs
% for i = 1:FileLength
% 
%     F = files{i};
%     [speech, fs] = audioread(F);
%     MFCCs = melcepst(speech, fs, 'Mtaz', NUM_MFCCs, 26);  
%     FemaleMFCCs = [FemaleMFCCs; MFCCs];
% end

    % Determine best fit GMM with AIC algorithm
% AIC = zeros(1, 5);
% GMModels = cell(1, 5);
% options = statset('MaxIter', 1000); 
% for k = 1:5
%     cInd = kmeans(FemaleMFCCs, k, 'Options', options, 'EmptyAction', 'singleton');
%     GMModels{k} = fitgmdist(FemaleMFCCs, k, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);
%     AIC(k)= GMModels{k}.AIC;
% end
% 
% [minAIC, numComponents] = min(AIC);
% BestModelFemale = GMModels{numComponents};


    % Fit GMM model to MFCCs
options = statset('MaxIter', 1000);         % limit max itterations without convergence

    % use kNN to initalise and set covariance type to diagonal
cInd = kmeans(FemaleMFCCs, NUM_MIXTURES, 'Options', options, 'EmptyAction', 'singleton');
BestModelFemale = fitgmdist(FemaleMFCCs, NUM_MIXTURES, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);

        

    %% Test Classifier
% TestDataRoot=[Root 'database/test/CLEAN_male/'];
% TestFeatureRoot=[Root 'features/test/'];
% testfiles=dir(TestDataRoot);
% testfiles=testfiles(3:end);
% features=dir(TestFeatureRoot);
% for n=3:length(features)
%     delete([TestFeatureRoot '/' features(n).name]);  
% end;


    
% FileLength = 450;
% FID = fopen('TestData.txt');
% FileLength = 300;
% FID = fopen('TestData2.txt');           % Read in test Data
% filenames = textscan(FID, '%s');
% fclose(FID);
% files = filenames{1};
% FemaleMFCCs = [];
TestDataRoot=[Root 'database/male/'];
% classification = cell(numfiles, 1);   % used to hold classifications
% numfiles=length(testfiles);
%round(numfiles_m*kk/10)
numfiles=round(numfiles_m*(kk+1)/num_fold) -round(numfiles_m*kk/num_fold);
%numfiles=numfiles_m*(9/10 + 1/10) -round(numfiles_m*9/10);
classification_m = cell(numfiles, 1);   % used to hold classifications
m=[];
p_m=[];
p_f=[];
f=[];
%for num=1:numfiles
index=1;
%for num= round(numfiles_m*9/10)+1 : numfiles_m*(9/10+1/10) 
for num= round(numfiles_m*kk/num_fold)+1 : round(numfiles_m*(kk+1)/num_fold)  
    %file_name=char(testfiles(num).name);
    file_name=char(testfiles_m(rand_m(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);  
    
%         % cepstral mean subtraction 
%     feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;
% 
%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;
    
    TestMFCCs = feature;
    % Calculate PDF for male and female GMMs
    ProbsMale = pdf(BestModelMale, TestMFCCs);          
    ProbsFemale = pdf(BestModelFemale, TestMFCCs);
    ProbsAll = ProbsMale+ ProbsFemale;
    prob_m = ProbsMale./ProbsAll;
    prob_f = ProbsFemale./ProbsAll;
    averageMale = mean(ProbsMale);
    ave_prob_m = mean(prob_m);
    ave_prob_f = mean(prob_f);
    averageFemale = mean(ProbsFemale);
    m=[m;averageMale];
    f=[f;averageFemale];
    p_m =[p_m; ave_prob_m];
    p_f=[p_f; ave_prob_f];
%     counterMale = 0;
%     counterFemale = 0;
%     
%     for j = 1:length(ProbsMale)
%         if (ProbsMale(j) > ProbsFemale(j))
%             counterMale = counterMale + 1;
%         else
%             counterFemale = counterFemale + 1;
%         end
%     end
    
    %if (averageMale > averageFemale)
    if (ave_prob_m > ave_prob_f)    
        classification_m{index} = 'M';
    else
        classification_m{index} = 'F';
    end
    index=index+1;
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;    

sum=0;
for i=1:length(classification_m)
    if (classification_m{i} == 'M')
        sum=sum+1;
    end
end
acc_m = sum/length(classification_m);
ave_acc_m=[ave_acc_m;acc_m];

TestDataRoot=[Root 'database/female/'];
numfiles=round(numfiles_f*(kk+1)/num_fold) -round(numfiles_f*kk/num_fold);
classification_f = cell(numfiles, 1);   % used to hold classifications
m=[];
p_m=[];
p_f=[];
f=[];
%for num=1:numfiles
index=1;
for num= round(numfiles_f*kk/num_fold)+1 : round(numfiles_f*(kk+1)/num_fold)   
    %file_name=char(testfiles(num).name);
    file_name=char(testfiles_f(rand_f(num)).name);
    wavFile=[TestDataRoot file_name];
    data=open_wavfile(wavFile);
    %feature=PNCC(data,Fs);
    feature=MFCC(data,Fs);  
    
%         % cepstral mean subtraction 
%     feature = feature - mean(feature, 2) * ones(1, size(feature, 2));
%     % cepstral mean normalization
%     std_mfcc = std(feature');
%     std_mfcc = std_mfcc';
%     feature = feature./std_mfcc;

%     % normalization for each frame
%     m_feature2 = mean(feature);
%     std_feature2 = std(feature);
%     feature = (feature - m_feature2)./std_feature2;
    
    
    TestMFCCs = feature;
    % Calculate PDF for male and female GMMs
    ProbsMale = pdf(BestModelMale, TestMFCCs);          
    ProbsFemale = pdf(BestModelFemale, TestMFCCs);
    ProbsAll = ProbsMale+ ProbsFemale;
    prob_m = ProbsMale./ProbsAll;
    prob_f = ProbsFemale./ProbsAll;
    averageMale = mean(ProbsMale);
    ave_prob_m = mean(prob_m);
    ave_prob_f = mean(prob_f);
    averageFemale = mean(ProbsFemale);
    m=[m;averageMale];
    f=[f;averageFemale];
    p_m =[p_m; ave_prob_m];
    p_f=[p_f; ave_prob_f];
%     counterMale = 0;
%     counterFemale = 0;
%     
%     for j = 1:length(ProbsMale)
%         if (ProbsMale(j) > ProbsFemale(j))
%             counterMale = counterMale + 1;
%         else
%             counterFemale = counterFemale + 1;
%         end
%     end
    
    %if (averageMale > averageFemale)
    if (ave_prob_m > ave_prob_f)    
        classification_f{index} = 'M';
    else
        classification_f{index} = 'F';
    end
    index=index+1;
    %feature_file=[TestFeatureRoot file_name(1:end-3) 'mfc'];
    %writehtk(feature_file,feature,1/120,9);
end;    

sum=0;
for i=1:length(classification_f)
    if (classification_f{i} == 'F')
        sum=sum+1;
    end
end
acc_f = sum/length(classification_f);
ave_acc_f=[ave_acc_f;acc_f];
end

ave_acc_m_final=mean(ave_acc_m);
ave_acc_f_final=mean(ave_acc_f);
% for i = 1:FileLength
% 
%     F = files{i};
%     [speech, fs] = audioread(F);
%     MFCCs = melcepst(speech, fs, 'Mtaz', NUM_MFCCs, 26);    % Get MFCCs of classified data
% 
%         % Calculate PDF for male and female GMMs
%     ProbsMale = pdf(BestModelMale, MFCCs);          
%     ProbsFemale = pdf(BestModelFemale, MFCCs);
% 
%         % Calculate average Liklihood of male or female classification
%         % based of PDF for each MFCCs section
%     averageMale = mean(ProbsMale);
%     averageFemale = mean(ProbsFemale);
%     
%     counterMale = 0;
%     counterFemale = 0;
%     
%     for j = 1:length(ProbsMale)
%         if (ProbsMale(j) > ProbsFemale(j))
%             counterMale = counterMale + 1;
%         else
%             counterFemale = counterFemale + 1;
%         end
%     end
% 
%     if (counterMale > counterFemale)
%         classification{i} = 'M';
%     else
%         classification{i} = 'F';
%     end
% 
%     if (classification{i} ~= Labels{i})
%         IncorrectCount = IncorrectCount + 1;
%         if (Labels{i} == 'F')
%             IncorrectFemale = IncorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             IncorrectMale = IncorrectMale + 1;
%         end
%         
%     else
%         if (Labels{i} == 'F')
%             CorrectFemale = CorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             CorrectMale = CorrectMale + 1;
%         end
%     end
%         % count number of correcta nd incorrect classifications
%         % Classification is male if average PDF of male GMM is greater the
%         % female PDF calculated at loacations given by MFCCs for each test
%         % utterance
% %     if (averageMale > averageFemale)
% %         classification{i} = 'M';
% %     else
% %         classification{i} = 'F';
% %     end
%     
%     if (counterMale > counterFemale)
%         classification{i} = 'M';
%     else
%         classification{i} = 'F';
%     end
% 
%     if (classification{i} ~= Labels{i})
%         IncorrectCount = IncorrectCount + 1;
%         if (Labels{i} == 'F')
%             IncorrectFemale = IncorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             IncorrectMale = IncorrectMale + 1;
%         end
%         
%     else
%         if (Labels{i} == 'F')
%             CorrectFemale = CorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             CorrectMale = CorrectMale + 1;
%         end
%     end
% end

% Precentage = ((FileLength - IncorrectCount)/FileLength)*100;        % final classification precentage
% 
% A1 = [CorrectMale, CorrectFemale, IncorrectMale, IncorrectFemale, Precentage];
% 
%     % Print Results to a files
% fileID = fopen('Results.txt','w');
% fprintf(fileID, 'Number of Correctly Identified Male Speakers %8.3f     \n', A1(1));
% fprintf(fileID, 'Number of Correctly Identified Female Speakers %8.3f   \n', A1(2));
% fprintf(fileID, 'Number of Incorrectly Identified Male Speakers %8.3f   \n', A1(3));
% fprintf(fileID, 'Number of Incorrectly Identified Female Speakers %8.3f \n', A1(4));
% fprintf(fileID, 'Total Precentage of Correct Classification %8.3f       \n', A1(5));