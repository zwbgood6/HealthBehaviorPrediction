# Health Behavior Prediction

[CIS520](https://alliance.seas.upenn.edu/~cis520/dynamic/2018/wiki/index.php?n=Main.HomePage) Machine Learning taught by [Lyle Ungar](http://www.cis.upenn.edu/~ungar/)

## Project

Predict health behaviors and outcomes for US counties

Team members: Wenbo Zhang, Yufu Wang, Zheyuan Xie

## File introduction

- Project slide: final_project_2018.zip

- Project report: cis520_project_final_report.pdf

- Dataset: CrossValidationDataset

- Matlab files:
predict_labels.m,
predict_labels_discriminative.m,
predict_labels_generative.m,
predict_labels_instance.m

## Models

There are a total of 4 models, all of which accept (train_inputs, train_labels, test_inputs), perform training and prediction. 

- predict_labels.m

This file contains the model we used for leaderboard submission and is our most accurate model. It used step-wise selected demo/SES features, as well as 70 principle components from the 2000 tweet features. The final regression model used Gaussian process with Matern 3/2 kernel. (valid_accuracy: 0.0713)

- predict_labels_generative.m

This file contains a model that makes use of generative method. It applies PCA as generative method to extract 70 principle components from 2000 tweet features. Then it used OLR with demo/SES features and the 70 selected features to make prediction. (valid_accuracy: 0.0777)

- predict_labels_discriminative.m

This file contains a model that makes use of discriminative method. It applies random forest as discriminative method to select the most important 70 features from 2000 tweet features. Then it used OLR with demo/SES features and the 70 selected features to make prediction. (valid_accuracy: 0.0872)

- predict_labels_instance.m

This file contains a model that makes use of instance-based method. It applies k-nearest-neighbors to find the 10 nearest data points and takes the average as prediction for a new data point. (valid_accuracy: 0.1166)


