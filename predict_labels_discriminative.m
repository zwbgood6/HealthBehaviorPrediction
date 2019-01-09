function pred_labels=predict_labels_discriminative(train_inputs,train_labels,test_inputs)
% This function apply random forest as discriminative method for feature
% selection. It will select 70 out of 2000 topic features. Then ordinary 
% linear regression is used as final regression model.
%
% Input: 
%   train_inputs : nxp matrix that contain training data
%                  n:  number of training samples
%                  p:  number of features
%
%   train_labels : nxq matrix that contain labels for training data
%                  n:  number of samples
%                  q:  number of labels
%
%   test_inputs  : mxp matrix that contain testing data
%                  m:  number of testing samples
%                  p:  number of features
%

pred_labels=randn(size(test_inputs,1),size(train_labels,2));

X_train = train_inputs;
X_valid = test_inputs;
Y_train = train_labels;

% First stage: demo/SES feature selection
selected = [1:7, 9:16, 17:20];
sq   = [1, 3, 12, 13, 15, 16];
sqrt = [4, 6, 1];
X1_train = [X_train(:,selected), X_train(:,sq).^2, X_train(:,sqrt).^(1/2)];
X1_valid = [X_valid(:,selected), X_valid(:,sq).^2, X_valid(:,sqrt).^(1/2)];


% Second stage: take all 2000 feature; selection later
X2_train = X_train(:,22:end);
X2_valid = X_valid(:,22:end);


% For each label, use Random Forest to select of set of LDA topic features
for i = 1:9
     
    y_train = Y_train(:,i);
    
    % Feature selection
    t = templateTree('MaxNumSplits',1);
    ens = fitrensemble(X2_train, y_train, 'Method','Bag','Learners',t);
    imp = predictorImportance(ens);
    [~, I] = sort(imp, 'descend');
    
    % Feature Fusion for this label
    X2imp_train = normalize(X2_train(:, I(1:70)));
    X2imp_valid = normalize(X2_valid(:, I(1:70)));
    X_train = [X1_train, X2imp_train];
    X_valid = [X1_valid, X2imp_valid];

    % Ordinary linear regression
    w = X_train \ y_train;
    
    % Make prediction
    pred_labels(:,i) = X_valid * w;
    
end


end

