clear; clc; close all;
%loading data
load('logisticModel.mat', 'mdl_log')
load('rfModel.mat', 'rf')
load('testSet.mat', 'test_tbl')
load('testPredictors.mat', 'Xtest')
load('ytrue_test.mat', 'ytrue_num')

%Logistic regression
ytrue_log = double(test_tbl.OutcomeCategory == "Good");
prob_log = predict(mdl_log, test_tbl);

%binary predictions
yhat_log = double(prob_log >= 0.5);
ytrue= double(test_tbl.OutcomeCategory== "Good");
yhat_log= double(prob_log>= 0.5);      
%confusion matrix
C_log = confusionmat(ytrue_log, yhat_log);
disp('Logistic regression confusion matrix:')
disp(C_log)
C= confusionmat(ytrue, yhat_log);
disp(C)

%metrics
accuracy_log = sum(yhat_log == ytrue_log) / numel(ytrue_log);
TP = C_log(2,2);
FP = C_log(1,2);
FN = C_log(2,1);
precision_log = TP / (TP + FP);
recall_log = TP / (TP + FN);
F1_log = 2 * (precision_log * recall_log) / (precision_log + recall_log);

fprintf('Logistic accuracy = %.3f\n', accuracy_log)
fprintf('Logistic precision (Good) = %.3f\n', precision_log)
fprintf('Logistic recall (Good) = %.3f\n', recall_log)
fprintf('Logistic F1-score (Good) = %.3f\n', F1_log)

%AUC
[~,~,~,AUC_log] = perfcurve(ytrue_log, prob_log, 1);
fprintf('Logistic AUC = %.3f\n\n', AUC_log)

%Random forest
%load test set
load('Xtest_.mat','Xtest')
[Ypred_rf, scores_rf] = predict(rf, Xtest);

class_names = rf.ClassNames;
idxGood = find(strcmp(class_names, 'Good'));
prob_rf = scores_rf(:, idxGood);

%binary pred
yhat_rf= categorical(Ypred_rf);
yhat_rf_num= double(yhat_rf== 'Good');  
 
%confusion matrix
C_rf= confusionmat(ytrue_num, yhat_rf_num);
fprintf('RF confusion matrix:\n'); 
disp(C_rf);
%metrics
accuracy_rf = sum(yhat_rf_num == ytrue_num) / numel(ytrue_num);

TP = C_rf(2,2);
FP = C_rf(1,2);
FN = C_rf(2,1);

precision_rf = TP / (TP + FP);
recall_rf = TP / (TP + FN);
F1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf);

fprintf('RF accuracy = %.3f\n', accuracy_rf)
fprintf('RF precision (Good) = %.3f\n', precision_rf)
fprintf('RF recall (Good) = %.3f\n', recall_rf)
fprintf('RF F1-score (Good) = %.3f\n', F1_rf)

%AUC
[~,~,~,AUC_rf] = perfcurve(ytrue_num, prob_rf, 1);
fprintf('RF AUC = %.3f\n', AUC_rf)
