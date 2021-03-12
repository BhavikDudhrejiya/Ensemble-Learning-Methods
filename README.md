# Ensemble-Learning-Methods

# Bagging

A. What is Bagging ?: 
1. It is an ensemble machine learning technique. 
2. It creates a tree based model on bootstrap sampling with replacement. 
3. Each vote are collected from the model and majority vote will be considered as a final output.
4. It is a bootstrap sampling + aggregation(aggragation of vote).
5. While creating model, Equal weights are assigned to each data.

B. Purpose of using Bagging:
It takes a models with high variance and low bias and reduce their variance without decreasing their bais.

C. What is Bootstrap ?:
1. Boostrap refers to a random sampling with replacement. 
2. It allows us to better understand the bias and the variance with the dataset. 
3. It involves random sampling of small subset of data from the dataset. This subset can be replace. 
4. The selection of all the example in the dataset has equal probability. 
5. This method can help to better understand the mean and standand deviation from the dataset.

D. What is Sample with Replacement ?:
Selecting same observation multiple time.

E. What is Aggregation ?:
Aggregate the outcome to each sample to estimate the most possible accurate statistics for overall sample.

F. What are the pro and cons of Bagging?: It reduce the overfitting problem. It means Bagging handles the data have high variance and low bais. Bagging is not helpful in case of high bais.

# Boosting

A. What is Boosting ?:
1. It is an ensemble machine learning technique.
2. It is a process that uses a set of machine learning algorithms to combine weak learners and form a strong learn in order to increase the accuracy of the model.
3. While creating a model, equal weights are assigned on first iteration later on weights are assigned on the performance on each iteration.

B. Purpose of Boosting ?:
It takes a models with high bias and low variance and reduce their bias without decreasing their variance.

C. Why Boosting is used ?:
Classic model build on the single or individual rule. Single rule mean creating model on a whole dataset. Single rules a not enough to make a strong prediction. It is also called weak rule or weak learner. Boosting technique helps us to create all weak learner and combine them to make it strong which predict better accuracy.

D. How Does Boosting Algorithm works ?:
1. We have a data.
2. Assigned equal weights to each data.
3. Iteration process will run untile all the misclassification converted into correctly classification:
   1. Random sampling on each iteration.
   2. Creating model on the random sampling on each iteration.
   3. Predicting on the model on each iteration.
   4. Identifying misclassication data.
   5. From second iteration, higher weights are assigning in each iteration on the data previously misclassified. 
   6. This process is run as much as possible to reduce misclassification error.
4.Testing the data on each model.
5.Predicting on the model.
6.Collecting the output.
7.Considering the majority vote as a final output.

E. Types of Boosting ?: Boosting has a three techniques:Adaboost, Gradient Boosting and XGBoosting
1. AdaBoost: 
2. Gradient Boosting
3. XGBoosting

F: How AdaBoosting is working ?
1. We have a dataset
2. Assigning equal weights to each observation
3. Finding best base learner
-Creating stumps or  base learners sequentially
-Computing Gini impurity or Entropy
-Whichever the learner have less impurity will be selecting as base learner
4. Train a model with base learner
5. Predicted on the model
6. Counting  Misclassification data
7. Computing Misclassification Error - Total error = sum(Weight of misclassified data)
8. Computing performance of the stumps - Performance of stumps = 1/2*Log-e(1-total error/total error)
9. Update the weights of incorrectly classified data - New Weight = Old Weight * Exp^performance of stump
    Updating the weights of correctly classified data - New Weight = Old Weight *  e^-performance of stump
10. Normalize the weight
11. Creating buckets on normalize weight
12. Algorithm generating random number equals to number of observations
13. Selecting where the random numbers fall in the buckets
14. Creating a new data
15. Running 2 to 14 steps above mentioned on each iteration until it each its limit
16. Prediction on the model with new data
17. Collecting votes from each base model
18. Majority vote will be considered as final output

# Gradient Boosting

How Gradient Boosting works ?
1. We have a Data.
2. Creating Base Learner.
3. Predicting Salary from base learner.
4. Computing loss function and extract residual.
5. Adding Sequential Decision Tree.
6. Predicting residual by giving experience and salary as predictors and residual as a target.
7. Predicting Salary from base learner prediction of salary and decision tree prediction of residual.
    - Salary Prediction = Base Learner Prediction + Learning Rate*Decision Tree Residual Prediction
    - Learning Rate will be in the range of 0 to 1
8. Computing loss function and extract residual.
9. Point 5 to 9 are a iterations. Each iteration decision tree will be added sequentially and prediction the salary.
   - Salary Prediction = Base Learner Prediction + Learning Rate*Decision Tree Residual Prediction1 + Learning Rate*Decision Tree Residual Prediction 2 + Learning Rate*Decision      Tree Residual Prediction...n
10. Testing the data - Testing data will be giving to the model which have minimum residual while prediction in iteration.
