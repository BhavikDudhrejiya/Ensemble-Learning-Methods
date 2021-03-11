# Ensemble-Learning-Methods

# Bagging

A. What is Bagging ?: 
1. It is an ensemble machine learning technique. 
2. It creates a tree based model on bootstrap sampling with replacement. 
3. Each vote are collected from the model and majority vote will be considered as a final output.
4. It is a bootstrap sampling + aggregation(aggragation of vote)

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

A. What is Boosting ?
1. It is an ensemble machine learning technique.
2. It is a process that uses a set of machine learning algorithms to combine weak learners and form a strong learn in order to increase the accuracy of the model.

B. Purpose of Boosting ?
It takes a models with high bias and low variance and reduce their bias without decreasing their variance.

C. Why Boosting is used ?
Classic model build on the single or individual rule. Single rule mean creating model on a whole dataset. Single rules a not enough to make a strong prediction. It is also called weak rule or weak learner. Boosting technique helps us to create all weak learner and combine them to make it strong which predict better accuracy.

D. How Does Boosting Algorithm works ?
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
   11. 
