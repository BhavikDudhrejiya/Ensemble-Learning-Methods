# Ensemble-Learning

**What is Ensemble Learning ?**
- Ensemble Learning is a method that is used to enhance the performance of machine learning model by combining several leaners. 
- When compared to a single model, this type of learning builds models with improved efficiency and accuracy.

**Types of Ensemble Learning**
- Bagging (parallel processing)
- Boosting (It is a Sequential processing)
- Stacking & Blending
- Voting Classifier

**Types of Ensembling**
- **Averaging**: Taking the average of predictions from models in case of regression problem or while predicting probabilities for the classification problem.
- **Majority Vote**: Taking the prediction with maximum vote / recommendation from multiple models predictions while predicting the outcomes of a classification problem.
- **Weighted Average**: In this, different weights are applied to predictions from multiple models then taking the average which means giving high or low importance to specific model output.

**Types of Bagging**
- Random forest
- Bagging meta-estimator

**Types of Boosting**
- AdaBoost
- GBM
- XGBM
- Light GBM
- CatBoost

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bagging

**What is Bagging ?**
- It is an ensemble machine learning technique. 
- It creates a tree based model on bootstrap sampling with replacement. 
- Each vote are collected from the model and majority vote will be considered as a final output.
- It is a bootstrap sampling + aggregation(aggragation of vote).
- While creating model, Equal weights are assigned to each data.

**Purpose of using Bagging**
- It takes a models with high variance and low bias and reduce their variance without decreasing their bais.

**What is Bootstrap ?**
- Boostrap refers to a random sampling with replacement. 
- It allows us to better understand the bias and the variance with the dataset. 
- It involves random sampling of small subset of data from the dataset. This subset can be replace. 
- The selection of all the example in the dataset has equal probability. 
- This method can help to better understand the mean and standand deviation from the dataset.

**What is Sample with Replacement ?**
- Selecting same observation multiple time.

**What is Aggregation ?**
- Aggregate the outcome to each sample to estimate the most possible accurate statistics for overall sample.

**What are the pro and cons of Bagging?**
- It reduce the overfitting problem. 
- It means Bagging handles the data have high variance and low bais. 
- Bagging is not helpful in case of high bais.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Boosting

**What is Boosting ?**
- It is an ensemble machine learning technique.
- It is a process that uses a set of machine learning algorithms to combine weak learners and form a strong learn in order to increase the accuracy of the model.
- While creating a model, equal weights are assigned on first iteration later on weights are assigned on the performance on each iteration.

**Purpose of Boosting ?**
- It takes a models with high bias and low variance and reduce their bias without decreasing their variance.

**Why Boosting is used ?**
- Classic model build on the single or individual rule. 
- Single rule mean creating model on a whole dataset. 
- Single rules a not enough to make a strong prediction. 
- It is also called weak rule or weak learner. 
- Boosting technique helps us to create all weak learner and combine them to make it strong which predict better accuracy.

**How Does Boosting Algorithm works ?**
- We have a data.
- Assigned equal weights to each data.
- Iteration process will run untile all the misclassification converted into correctly classification:
  - Random sampling on each iteration.
  - Creating model on the random sampling on each iteration.
  - Predicting on the model on each iteration.
  - Identifying misclassication data.
  - From second iteration, higher weights are assigning in each iteration on the data previously misclassified. 
  - This process is run as much as possible to reduce misclassification error.
- Testing the data on each model.
- Predicting on the model.
- Collecting the output.
- Considering the majority vote as a final output.

**Types of Boosting ?** 
Boosting has a three techniques
- AdaBoost
- Gradient Boosting
- XGBoosting
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AdaBoosting

**How AdaBoosting is working ?**
- We have a dataset.
- Assigning equal weights to each observation.
- Finding best base learner.
- Creating stumps or  base learners sequentially.
- Computing Gini impurity or Entropy.
- Whichever the learner have less impurity will be selecting as base learner.
- Train a model with base learner.
- Predicted on the model.
- Counting  Misclassification data.
- Computing Misclassification Error - Total error = sum(Weight of misclassified data).
- Computing performance of the stumps - Performance of stumps = 1/2*Log-e(1-total error/total error).
- Update the weights of incorrectly classified data - New Weight = Old Weight * Exp^performance of stump.
- Updating the weights of correctly classified data - New Weight = Old Weight *  e^-performance of stump.
- Normalize the weight.
- Creating buckets on normalize weight.
- Algorithm generating random number equals to number of observations.
- Selecting where the random numbers fall in the buckets.
- Creating a new data.
- Running 2 to 14 steps above mentioned on each iteration until it each its limit.
- Prediction on the model with new data.
- Collecting votes from each base model.
- Majority vote will be considered as final output.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Gradient Boosting

**How Gradient Boosting works ?**
- We have a Data.
- Creating Base Learner.
- Predicting Salary from base learner.
- Computing loss function and extract residual.
- Adding Sequential Decision Tree.
- Predicting residual by giving experience and salary as predictors and residual as a target.
- Predicting Salary from base learner prediction of salary and decision tree prediction of residual.
    - Salary Prediction = Base Learner Prediction + Learning Rate*Decision Tree Residual Prediction
    - Learning Rate will be in the range of 0 to 1.
- Computing loss function and extract residual.
- Point 5 to 9 are a iterations. Each iteration decision tree will be added sequentially and prediction the salary.
   - Salary Prediction = Base Learner Prediction + Learning Rate*Decision Tree Residual Prediction1 + Learning Rate*Decision Tree Residual Prediction 2 + Learning Rate*Decision      Tree Residual Prediction...n.
- Testing the data - Testing data will be giving to the model which have minimum residual while prediction in iteration.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Extreme Gradient Boosting

**What is Extreme Graddient Boosting ?**
- It is an open source software library which provides a gradient boosting frame work. 
- It is implementing gradient boosting decision tree algorithm.
- It is using gradient boosting algorithm to minimize the loss when adding a new model.
- It is using both in regression and classification problem.

**How Extreme Gradient Boosting works ?**
- We have a Data.
- Constructing Base Model.
- Base learner takes probability 0.5 & computing residual.
- Constructing Decision as per below:
    - Computing Similarity Weights: ∑(Residual)^2 / ∑P(1-P) + lambda.
    - Computing Similarity Weight of Root Node.
    - Computing Similarity Weight of left side decision node & its leaf node.
    - Computing Similarity Weight of right side decision node & its leaf node.
    - Computing Gain = Leaf1 Similarity W + Leaf2 Similarity W - Root Node Similarity Weight.
    - Computing Gain of Root Node & left side of decision node and its leaf node.
    - Computing Gain of Root Node & right side of decision node and its leaf node.
    - Computing Gain of other combination of features of decision node and its leaf node.
    - Selecting the Root Node, Decision node and leaf node have high information gain.
- Predicting the probability = Sigmoid(log(odd) of Prediction of Base Learner + learning rate(Prediction of Decision Tree)).
- Predicting residual = Previous residual - Predicted Probability.
- Running the iteration from point 2 to 6 and at the end of the iteration, The residual will be the minimal.
- Test Prediction on the model of iteration have minimal residual.

**Advantage of XGBoost**
- XGBoost has inbuult regularization of L1(Lasso) & L2(Ridge) to prevent overfitting.
- Providing parallel processing due to which it become a much faster than Gradient Boosting.
- Handling missing values.
- It allows the user to run cross validation at each iteration.
- Effective tree pruning by stop spliting while encounter negative loss.
