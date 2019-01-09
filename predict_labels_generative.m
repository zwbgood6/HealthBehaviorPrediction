function pred_labels=predict_labels_generative(train_inputs,train_labels,test_inputs)
% This function apply PCA as generative method to extract new features from
% the 2000 LDA topic features. Then ordinary linear regression is used as
% final regression model.
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

X_train = train_inputs;
X_valid = test_inputs;
Y_train = train_labels;

% First stage: demo/SES feature selection
selected = [1:7, 9:16, 17:20];
sq   = [1, 3, 12, 13, 15, 16];
sqrt = [4, 6, 1];
X1_train = [X_train(:,selected), X_train(:,sq).^2, X_train(:,sqrt).^(1/2)];
X1_valid = [X_valid(:,selected), X_valid(:,sq).^2, X_valid(:,sqrt).^(1/2)];


% Second stage: PCA as generative method to extract new features
X2_train = X_train(:,22:end);
X2_valid = X_valid(:,22:end);

p = 70;
coeff = pca(X2_train);
coeff = coeff(:,1:p);
X2pca_train = (X2_train - mean(X2_train)) * coeff;
X2pca_valid = (X2_valid - mean(X2_valid)) * coeff;


% Feature Fusion
X_train = [X1_train,  X2pca_train];
X_valid = [X1_valid,  X2pca_valid ];


% Ordinary linear regression
W = X_train \ Y_train;


% Make prediction
pred_labels = X_valid * W;



end

