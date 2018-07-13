%Mode 설명 
%아래와 같은 모드를 입력해야만 올바르게 코드가 동작합니다.
%총 설정을 해줘야 하는 mode는 4가지 입니다. 

%   1. mode_code : train or test
%   2. mode_object : human or face
%   3. mode_feature : HOG or LBP
%   4. mode_cascade : true or false

%   코드 동작 예시 : 
%       (1). face data에 대해 HOG feature로 cascade 안쓰고 학습시키고 싶을 경우 
%           >>clear
%           >>mode_code='train';
%           >>mode_obect='face';
%           >>mode_feature='HOG';
%           >>mode_cascade=false;
%           >>main_script;
%
%       (2). (1)와 같은 mode 의 model을 테스트 하고 싶을 경우
%           >>clear
%           >>mode_code='test';
%           >>main_script;


%   코드 동작시 주의사항 :  1. 학습을 반드시 먼저 해야함, 학습을 하면 학습한 환경이 'model.dat'에 저장되며 
%                        2. mode_code 를 test로 하여 돌리면 'model.dat'에 저장된 mode로 자동 설정되어 돌아감 
%                        3. mode 설정을 제대로 안하면 버그뜸 




%start
%mode_code : 
%mode_object : 
%mode_feature :
%mode_cascade :


addpath('./utils/');
addpath('./svm/');
addpath('./svm/minFunc/');
addpath('./human/');
addpath('./face/');


if strcmp(mode_code, 'train')
    fprintf('code : train\n');

    if strcmp(mode_feature, 'HOG')
        f.feature='hog';
        fprintf('feature : HOG\n');
        
    elseif strcmp(mode_feature,'LBP')
        f.feature='lbp';
        fprintf('feature : LBP\n');
    else
        f.feature='mix';
        fprintf('you are wrong(decide mode_feature)\n');
    end
  
    if mode_cascade==true
        fprintf('cascade : true\n');
        f.feature=true;
    elseif mode_cascade==false
        fprintf('cascade : false\n');
        f.feature=false;
    else
        fprintf('you are wrong(decide mode_cascade)\n');
    end 

    if strcmp(mode_object, 'face')
        fprintf('object : face\n');
        f.object='face';
        f.feature=mode_feature;
        f.cascade=mode_cascade;
        train_face(f);
    elseif strcmp(mode_object,'human')
        fprintf('object : human\n');
        f.object='human';
        f.feature=mode_feature;
        f.cascade=mode_cascade;
        train_human(f);
    else
        fprintf('you are wrong(decide mode_object)\n');
    end



elseif strcmp(mode_code,'test')
    fprintf('code : test\n');
    load('model.mat','f');
    if strcmp(f.object,'face')
        fprintf('object : face\n');
        test_face(f);
    else
        fprintf('object : human\n');
        test_human(f);
    end


else
    fprintf('you are wrong(decide mode_code\n');
end





