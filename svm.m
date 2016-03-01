function [predict_lbl,model] = svm(trn_ftr,trn_lbl)

addpath('./libsvm-3.20/matlab');

h1 = size(trn_ftr,1);
randorder1 = randperm(h1);
rand_ftr = trn_ftr(randorder1, :);
rand_lbl = trn_lbl(randorder1, :);

% Unbalanced Class
%       l1 = sum(rand_label == 0); 
%       l2 = sum(rand_label == 1);
%       
%       w1 = 100*(l2/(l1+l2));  
%       w2 = 100*(l1/(l1+l2));
% 
%       cmd = ['-q -s 2 -v 10 -c ', num2str(10^log2c) ' -w-1 '  num2str(w1) ' -w1 ' num2str(w2)];

% TRAIN
bestcv = 0;
for log2c = -5:5,
    for log2g = -5:5,
        cmd = ['-t 0 -v 5 -c ', num2str(10^log2c), ' -g ', num2str(10^log2g)];
        cv = svmtrain(rand_lbl, rand_ftr, cmd);
        if (cv >= bestcv),
            bestcv = cv; 
            bestc = 10^log2c; 
            bestg = 10^log2g;
        end
    end
end

clc
fprintf('(best c=%g, g=%g, rate=%g)\n', bestc, bestg, bestcv);

cmd = ['-t 0 -c ', num2str(bestc), ' -g ', num2str(bestg)];

model = svmtrain(rand_lbl, rand_ftr, cmd);
 
% TEST
[predict_lbl, accuracy, dec_val] = svmpredict(trn_lbl, trn_ftr, model);

