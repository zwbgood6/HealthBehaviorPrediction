function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

pred_labels=randn(size(test_inputs,1),size(train_labels,2));

X_train = train_inputs;
X_valid = test_inputs;

% First stage: demo/SES feature selection
selected = [1:7, 9:16, 17:20];
sq   = [1, 3, 12, 13, 15, 16];
sqrt = [4, 6, 1];
X1_train = [X_train(:,selected), X_train(:,sq).^2, X_train(:,sqrt).^(1/2)];
X1_valid = [X_valid(:,selected), X_valid(:,sq).^2, X_valid(:,sqrt).^(1/2)];


% Second stage: LDA topics reduction
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

pred_train = randn(size(train_inputs,1),size(train_labels,2));
    
% Train 9 seperate Gaussian Process for each labels, and predict
for i = [5, 7, 6, 4, 9, 1,2,3,8]
    
    y_train = train_labels(:,i);
    
    gpr = fitrgp(X_train, y_train,'FitMethod', 'sd', 'Standardize',1, ...
                'Basis','constant','KernelFunction','matern32', ...
                'Optimizer', 'fminunc');
    
    pred_train(:,i)  = predict(gpr, X_train);
    pred_labels(:,i) = predict(gpr, X_valid);
    
    X_train = [pred_train(:,i).^2, X_train];
    X_valid = [pred_labels(:,i).^2, X_valid];
    
end


end

