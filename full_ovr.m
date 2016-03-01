function [model] = full_ovr(y, x)

clc

% addpath to the libsvm toolbox
addpath('./libsvm-3.20/matlab');

% Get Train and Test Data

% TRAIN DATA
dnaTrainData = x;
dnaTrainLabel = y;
NTrain = size(dnaTrainData,1);

% Randomize and then sort
randorder1 = randperm(NTrain);
dnaTrainData = dnaTrainData(randorder1, :);
dnaTrainLabel = dnaTrainLabel(randorder1, :);

[dnaTrainLabel, permIndex] = sortrows(dnaTrainLabel);
dnaTrainData = dnaTrainData(permIndex,:);

%TEST DATA
dnaTestData = x;
dnaTestLabel = y;
NTest = size(dnaTestData,1);

% Randomize and then sort
randorder1 = randperm(NTrain);
dnaTestData = dnaTestData(randorder1, :);
dnaTestLabel = dnaTestLabel(randorder1, :);

[dnaTestLabel, permIndex] = sortrows(dnaTestLabel);
dnaTestData = dnaTestData(permIndex,:);

% Combine the data together to fit the format
totalData = [dnaTrainData; dnaTestData];
totalLabel = [dnaTrainLabel; dnaTestLabel];

[N D] = size(totalData);
labelList = unique(totalLabel(:));
NClass = length(labelList);

% #######################
% Determine the train and test index
% #######################
trainIndex = zeros(N,1); trainIndex(1:NTrain) = 1;
testIndex = zeros(N,1); testIndex( (NTrain+1):N) = 1;
trainData = totalData(trainIndex==1,:);
trainLabel = totalLabel(trainIndex==1,:);
testData = totalData(testIndex==1,:);
testLabel = totalLabel(testIndex==1,:);

% #######################
% Parameter selection using 3-fold cross validation
% #######################

bestcv = 0;
for log2c = -1:1:3,
  for log2g = -4:1:2,
    cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = get_cv_ac(trainLabel, trainData, cmd, 3);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end

% #######################
% Train the SVM in one-vs-rest (OVR) mode
% #######################

bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = ovrtrain(trainLabel, trainData, bestParam);


% % #######################
% % Classify samples using OVR model
% % #######################
% [predict_label, accuracy, prob_values] = ovrpredict(testLabel, testData, model);
% fprintf('Accuracy = %g%%\n', accuracy * 100);



