function pred_labels=predict_labels_instance(train_inputs,train_labels,test_inputs)
% This function apple k-nearest-neighbors and use the average as regression
% prediction.
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

% KNN
Idx = knnsearch(X_train, X_valid, 'K', 10);

% Make prediction
for i = 1:size(X_valid, 1)
    
    knn = Y_train(Idx(i,:), :);
    pred_labels(i, :) = mean(knn);
    
end


end

