% train_clean_test_clean
doDisp=1;

Fs=8000;

Root='./';

TrainDataRoot=[Root 'database/train/'];
TestDataRoot=[Root 'database/test/CLEAN/'];

TrainFeatureRoot=[Root 'features/train/'];
TestFeatureRoot=[Root 'features/test/'];

ModelRoot=[Root 'models/'];
ScriptRoot=[Root 'scripts/'];
wdnet_file      = [ScriptRoot 'wdnet_file.txt'];

ConfigFile=[ScriptRoot 'config_file.cfg'];
ModelList=[ScriptRoot 'Model_List.txt'];
DictFile=[ScriptRoot 'Dictionary.txt'];
WordList=[ScriptRoot 'Word_List.txt'];
WordList2=[ScriptRoot 'Word_List2.txt'];
WordListSP=[Root 'scripts/WordListSP.txt'];
MLF_Results=[ScriptRoot 'MLF_Results.mlf'];
TrainWordMLF=[Root 'scripts/TrainWordMLF.mlf'];
TrainWordMLF2=[Root 'scripts/TrainWordMLF2.mlf'];
TestWordMLF=[Root 'scripts/TestWordMLF.mlf'];
TrainFeatureScript=[Root 'scripts/TrainFeatureScript.txt'];
TestFeatureScript=[Root 'scripts/TestFeatureScript.txt'];
TestScript=[Root 'scripts/TestScript.txt'];
MixScript1=[Root 'scripts/HED1.txt'];
MixScript2=[Root 'scripts/HED2.txt'];
WdnetFile=[Root 'scripts/WDNet.txt'];
MLFResults=[Root 'scripts/MLFResults.mlf'];
hmmdefs=[ModelRoot 'hmmdefs'];


%bb=load('BestModelMale.mat');
%gender_identify_GMM_MFCC_3
disp('>>> Gender Identifying Model Trained !');

NUM_STATES=16;
NUM_HEREST_1=3;
NUM_HEREST_2=6;
alpha=1.16;
%=======================================================================================
%=============== Training Feature Extraction ===========================================
%=======================================================================================
trainfiles=dir([TrainDataRoot]);
trainfiles=trainfiles(3:end);
features=dir(TrainFeatureRoot);
for n=3:length(features)
    delete([TrainFeatureRoot '/' features(n).name]);  
end;
if(doDisp)

    disp('>>> Performing training feature extraction');

end;     
number=length(trainfiles);
for num=1:number
    if(doDisp & mod(num,200)==0)
        disp([num2str(ceil(100*num/number)) '% done...']);
    end;
    file_name=char(trainfiles(num).name);
    wavFile=[TrainDataRoot file_name];
    data=open_wavfile(wavFile);
    
    
    %feature=PNCC(data,Fs);
     feature=MFCC(data,Fs);
     %feature=PLP(data,Fs);
     %feature=PCA(data,Fs);
%     %feature=PNCC_warp_freq_low(data,Fs, alpha);
%     
%     
%     
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
%     m=[m;averageMale];
%     f=[f;averageFemale];
%     p_m =[p_m; ave_prob_m];
%     p_f=[p_f; ave_prob_f];
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
        %classification_m{index} = 'M';
        feature=PNCC3(data,Fs);
    else
        %classification_m{index} = 'F';
        feature=PNCC3_warp(data,Fs);
    end
    
% %     
% %     
% %     
    
    
    
    feature_file=[TrainFeatureRoot file_name(1:end-3) 'mfc'];
    writehtk(feature_file,feature,0.01,9);
end;
fid=fopen(TrainFeatureScript,'w');
for i=1:length(trainfiles)
    fprintf(fid, '%s\n',[TrainFeatureRoot trainfiles(i).name(1:end-3) 'mfc']);
end
fclose(fid);

if(doDisp)

    disp('>>> Training HMMs');

end;
feature_files=char(textread(TrainFeatureScript,'%s'));
fid1=fopen(TrainWordMLF,'w');
fid2=fopen(TrainWordMLF2,'w');
fprintf(fid1,'%s\n','#!MLF!#');
fprintf(fid2,'%s\n','#!MLF!#');
for k=1:size(feature_files,1)
    dashes=find(feature_files(k,:)=='-');
    dots=find(feature_files(k,:)=='.');
    slashes=find(feature_files(k,:)=='\');
    underscores=find(feature_files(k,:)=='_');
    for s=1:length(slashes)
        feature_files(k,slashes(s))='/';
    end;
    fprintf(fid1,'%s\n',['"' feature_files(k,1:dots(end)) 'lab"']);
    fprintf(fid1,'%s\n','sil');
    fprintf(fid2,'%s\n',['"' feature_files(k,1:dots(end)) 'lab"']);
    fprintf(fid2,'%s\n','sil');
    words=feature_files(k,underscores(end)+1:dots(end)-2);
    for w=1:length(words)
        number=find_number(words(w));
        fprintf(fid1,'%s\n',number);
        fprintf(fid2,'%s\n',number);
        if(w<length(words))
            fprintf(fid2,'%s\n','sp');
        end;
    end;
    fprintf(fid1,'%s\n','sil');
    fprintf(fid1,'%s\n','.');
    fprintf(fid2,'%s\n','sil');
    fprintf(fid2,'%s\n','.');
end;
fclose(fid1);
fclose(fid2);
makeProto(ModelRoot,'IHMM','USER',size(feature,2),NUM_STATES);
disp('HCompV.');
cmd = ['!HCompV '...
        ' -T 2 '...
        ' -D '...
        ' -C ' ConfigFile...
        ' -f 0.01 -m'...
        ' -S ' TrainFeatureScript...
        ' -M ' ModelRoot...
        ' '    [ModelRoot 'proto']...
        ];
eval(cmd);
models=textread(WordList,'%s','delimiter','\n');  
hmm_files=dir(ModelRoot);
for k=1:length(models);
    model_hmm=[ModelRoot char(models(k))];
    hmmdef=textread([ModelRoot 'proto'],'%s', 'delimiter', '\n');
    fid=fopen(model_hmm,'w');
    for line=1:3
        fprintf(fid,'%s\n',char(hmmdef(line)));
    end;
    fprintf(fid,'%s\n',['~h "' char(models(k)) '"']);
    for line=5:length(hmmdef)
        fprintf(fid,'%s\n',char(hmmdef(line)));
    end;
    fclose(fid);
end;  
hmm_files=dir(ModelRoot);
fid=fopen(hmmdefs,'w');
for mdl=1:length(models)
    curr_mdl=char(models{mdl});
    phn_hmm=[ModelRoot char(models{mdl})];
    lines=textread([ModelRoot char(models{mdl})], '%s', 'delimiter', '\n');
    for k=1:length(lines)
        fprintf(fid,'%s\n',lines{k});
    end;
    fprintf(fid,'%s\n');
end;
fclose(fid);   
for N=1:NUM_HEREST_1
    disp(['HERest (iteration ' num2str(N) ').']);
    cmd=['! HERest'...
                ' -D '...
                ' -C ' ConfigFile...
                ' -I ' TrainWordMLF...
                ' -t 250.0 150.0 1000.0'...
                ' -S ' TrainFeatureScript...
                ' -H ' hmmdefs...
                ' -M ' ModelRoot... 
                ' ' WordList ...
                ];
    eval(cmd);
end;
disp(['HHEd: 1.']);
cmd=['!HHEd'...
        ' -H ' hmmdefs...
        ' ' MixScript1...
        ' ' WordList...
        ];
eval(cmd);
for N=1:NUM_HEREST_1
        disp(['HERest (iteration ' num2str(N) ').']);
        cmd=['! HERest'...
            ' -D '...
            ' -C ' ConfigFile...
            ' -I ' TrainWordMLF...
            ' -t 250.0 150.0 1000.0'...
            ' -S ' TrainFeatureScript...
            ' -H ' hmmdefs...
            ' -M ' ModelRoot... 
            ' ' WordList...
            ];
eval(cmd);
end;        
disp(['HHEd: 2.']);
cmd=['!HHEd'...
        ' -H ' hmmdefs...
        ' ' MixScript2...
        ' ' WordList...
        ];
eval(cmd);
fix_hmmdefs(size(feature,2),Root);
for N=1:NUM_HEREST_2 
        disp(['HERest (iteration ' num2str(N) ').']);
        cmd=['! HERest'...
            ' -D '...
            ' -C ' ConfigFile...
            ' -I ' TrainWordMLF2...
            ' -t 250.0 150.0 1000.0'...
            ' -S ' TrainFeatureScript...
            ' -H ' hmmdefs...
            ' -M ' ModelRoot... 
            ' ' WordListSP...
            ];
        eval(cmd);
end; 

