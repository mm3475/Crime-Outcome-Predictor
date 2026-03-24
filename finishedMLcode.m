data= readtable("UK_Crime_Oct_Dec_2022.csv");
data.Properties.VariableNames;
head(data);
summary(data);

%outcomes are text so convert to string
opts= detectImportOptions('UK_Crime_Oct_Dec_2022.csv');
opts= setvartype(opts, 'outcome', 'string');
data= readtable('UK_Crime_Oct_Dec_2022.csv', opts);

%conv to string and remove missing
outcome_str= string(data.outcome);
validIdx= ~ismissing(outcome_str) & strlength(strtrim(outcome_str))> 0& outcome_str~= "NaN";
outcome_str= outcome_str(validIdx);
%find unique outcomes
outcome_crime= unique(outcome_str);
disp(outcome_crime);

%convert outcome to str and to binary
outcome_str= string(data.outcome);
missing_outcome= ismissing(outcome_str) | strlength(strtrim(outcome_str))==0;
data= data(~missing_outcome,:);
outcome_str= outcome_str(~missing_outcome);

neutral= ["Action to be taken by another organisation",
           "Awaiting court outcome",
           "Court result unavailable",
           'Status update unavailable',
           "Suspect charged as part of another case"];

data= data(~ismember(data.outcome, neutral), :);

good_words= ["Local", "caution", "drugs", "penalty", 'outcome'];

good_binary= contains(data.outcome, good_words, "IgnoreCase", true);
data.OutcomeCategory= repmat("Bad", height(data), 1);
data.OutcomeCategory(good_binary)= "Good";
data.OutcomeBinary= double(data.OutcomeCategory=="Good");

good_outcomes= unique(outcome_str(good_binary));
%check how many outcomes are good/bad
num_goodoutcomes= sum(data.OutcomeBinary);
fprintf('Total good outcomes: %d\n', num_goodoutcomes);
num_badoutcomes= sum(data.OutcomeBinary== 0);
fprintf('Total bad outcomes: %d\n', num_badoutcomes);


%% preprocessing
%conv to categ varaibles
data.crime_type= categorical(string(data.crime_type));
%check frequ of crimes
crime_categ= categories(data.crime_type);   
crime_counts= countcats(data.crime_type); 
crime_summary= table(crime_categ, crime_counts, 'VariableNames', {'CrimeType','Count'});
crime_summary= sortrows(crime_summary, 'Count', 'descend');
disp(crime_summary);
%reduce noise by grouping into 'other'


%convert to string to avoid categorical overwrite issues
crime = string(data.crime_type);

%define category groups
violent= ["violent-crime", "robbery", "possession-of-weapons"];
property= ["other-theft", "vehicle-crime", "theft-from-the-person", ...
            "burglary", "shoplifting", "bicycle-theft"];
publicorder= ["public-order", "criminal-damage-arson"];
drugrelated= ["drugs"];

%new variable other
new_crime= repmat("Other", height(data), 1);

%assign categories
new_crime(ismember(crime, violent))= "Violent";
new_crime(ismember(crime, property))= "Property";
new_crime(ismember(crime, publicorder))= "PublicOrder";
new_crime(ismember(crime, drugrelated))= "Drugs";

%convert to categorical
data.crime_group = categorical(new_crime);

%repeat for force id
data.force_id= categorical(string(data.force_id));
force_categ= categories(data.force_id);
force_count= countcats(data.force_id);
force_summary= table(force_categ, force_count, 'VariableNames', {'ForceType','Count'});
force_summary= sortrows(force_summary, 'Count', 'descend');
disp(force_summary);

%training data 1
rng(1) % reproducible data
cvp= cvpartition(data.OutcomeBinary,'HoldOut', 0.3);  % split training and testing data 7 to 3
train_tbl= data(training(cvp), :); %extract training data
test_tbl= data(test(cvp), :); %extract test data
tabulate(train_tbl.OutcomeCategory) 
tabulate(test_tbl.OutcomeCategory)
%data is extremely unbalanced


%% so upsample minority 
%separate good/bad training set 
good_rows= train_tbl(train_tbl.OutcomeBinary == 1, :); 
bad_rows= train_tbl(train_tbl.OutcomeBinary == 0, :); 

%upsample minority class 
multiplier= floor(height(bad_rows) / height(good_rows)); 
remainder= mod(height(bad_rows), height(good_rows)); 
upsampled_good= [repmat(good_rows, max(multiplier, 1), 1); ...
    good_rows(1:remainder, :) ...
];
%Recombine balanced training set 
balanced_traintbl= [bad_rows; upsampled_good]; 
%shuffle rows 
balanced_traintbl= balanced_traintbl(randperm(height(balanced_traintbl)), :); 
%check again 
tabulate(balanced_traintbl.OutcomeCategory) 


%% logistic regression 
%convert to categ 
balanced_traintbl.crime_type = removecats(balanced_traintbl.crime_type);
balanced_traintbl.force_id = removecats(balanced_traintbl.force_id);
balanced_traintbl.OutcomeCategory = categorical(balanced_traintbl.OutcomeCategory, ["Bad","Good"]);
formula = 'OutcomeCategory ~ crime_group + force_id'; 
mdl_log = fitglm(balanced_traintbl, formula, 'Distribution', 'binomial');
disp(mdl_log) 
%save it
save('logisticModel.mat', 'mdl_log');


%% Predict on test set
prob_log= predict(mdl_log, test_tbl); 
yhat_log= prob_log>= 0.5;
ytrue= double(test_tbl.OutcomeCategory== "Good");
yhat_log= double(prob_log>= 0.5);      
C= confusionmat(ytrue, yhat_log);
disp(C)

%compute metrics
accuracy= sum(yhat_log== ytrue)/numel(ytrue);
fprintf('Logistic accuracy= %.3f\n', accuracy);

TP= C(2,2);  FP = C(1,2);  FN = C(2,1);
precision= TP / (TP + FP);
recall= TP / (TP + FN);
F1= 2* (precision* recall)/ (precision+ recall);

fprintf('Precision(Good)= %.3f\n', precision)
fprintf('Recall(Good)= %.3f\n', recall)
fprintf('F1-score(Good)= %.3f\n', F1)

[fp_log, tp_log, thr_log, AUC_log]= perfcurve(ytrue, prob_log, 1);
fprintf('Logistic AUC= %.3f\n', AUC_log);

%% Random forests code
%predictors table
predictor_names= {'crime_group','force_id'};
Xtrain= train_tbl(:, predictor_names);
Ytrain = categorical(train_tbl.OutcomeCategory);

%treebagger
num_trees= 200;  % number of trees in the Random Forest
rf = TreeBagger(num_trees, Xtrain, Ytrain, ...
    'Method','classification', ...
    'OOBPrediction','on', ...
    'OOBPredictorImportance','on', ...
    'NumPredictorsToSample','all', ...
    'Prior', 'uniform' ...      % treat classes equally
);

%out-of-bag error plot
figure; oobErrorBaggedEnsemble= oobError(rf);
plot(oobErrorBaggedEnsemble);
xlabel('Number of trees'); ylabel('OOB classification error');
title('Out-of-bag error');

save('rfModel.mat','rf');
%prediction on test set
Xtest = balanced_traintbl(:, predictor_names);
save('Xtest_.mat','Xtest')
[Ypred_rf, scores_rf] = predict(rf, Xtest);  % Ypred_rf are cell strings 'Good'/'Bad'
%find index:
class_names = rf.ClassNames;  % e.g. {'Bad','Good'}
idxGood = find(strcmp(class_names,'Good'));
prob_rf = scores_rf(:, idxGood);

ytrue_num= double(balanced_traintbl.OutcomeCategory== 'Good');  
save('ytrue_num.mat', 'ytrue_num');
%convert predicted outcomes to numeric
yhat_rf= categorical(Ypred_rf);  
yhat_rf_num= double(yhat_rf== 'Good');  

%confusion matrix
C_rf= confusionmat(ytrue_num, yhat_rf_num);
fprintf('RF confusion matrix:\n'); 
disp(C_rf);

%AUC
[fp_rf, tp_rf, thr_rf, AUC_rf]= perfcurve(ytrue_num, prob_rf, 1);
fprintf('RF AUC = %.3f\n', AUC_rf);

imp= rf.OOBPermutedPredictorDeltaError; %importance measure

%metrics
accuracy_rf= sum(yhat_rf_num== ytrue_num)/ numel(ytrue_num);
TP= C_rf(2,2);
FP= C_rf(1,2);
FN= C_rf(2,1);

precision_rf= TP/ (TP+ FP);
recall_rf= TP/ (TP+ FN);
F1_rf= 2* (precision_rf* recall_rf)/ (precision_rf+ recall_rf);

fprintf('RF accuracy= %.3f\n', accuracy_rf);
fprintf('Precision (Good)= %.3f\n', precision_rf);
fprintf('Recall (Good)= %.3f\n', recall_rf);
fprintf('F1-score (Good)= %.3f\n', F1_rf);
%% Graphs
%ROC Curve
ytrue_log= test_tbl.OutcomeCategory;     
ytrue_log= categorical(ytrue_log);

prob_log= predict(mdl_log, test_tbl);  

[fp_log, tp_log, ~, AUC_log]= perfcurve(ytrue_log, prob_log, 'Good');

figure;
plot(fp_log, tp_log, 'LineWidth', 2); hold on;
plot([0 1], [0 1], '--', 'Color', [1 0.4 0.6]);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('Logistic Regression ROC Curve (AUC = %.3f)', AUC_log));
grid on;

coef= mdl_log.Coefficients.Estimate;
coefNames= mdl_log.CoefficientNames;

figure;
bar(coef);
set(gca, 'XTickLabel', coefNames, 'XTickLabelRotation', 45);
ylabel('Coefficient Estimate');
title('Logistic Regression Coefficients');
grid on;

figure;
plot(fp_rf, tp_rf, 'LineWidth', 2); hold on;
plot([0 1], [0 1], '--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('Random Forest ROC Curve (AUC = %.3f)', AUC_rf));
grid on;

figure;
bar(imp);
set(gca,'XTickLabel', predictor_names, 'XTickLabelRotation', 45);
ylabel('OOB Permuted Predictor Delta Error');
title('Random Forest Predictor Importance');