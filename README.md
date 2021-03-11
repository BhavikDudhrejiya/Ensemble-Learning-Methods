# Ensemble-Learning-Methods

# Bagging

A. What is Bagging ?: 
1. It is an ensemble machine learning technique. 
2. It creates a tree based model on bootstrap sampling with replacement. 
3. Each vote are collected from the model and majority vote will be considered as a final output.
4. It is a bootstrap sampling + aggregation(aggragation of vote)

B. Purpose of using Bagging:
1. It takes a models with high variance and low bias and reduce their variance without decreasing their bais.

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

F. What are the pro and cons of Bagging?: It reduce the overfitting problem. It means Bagging handle the data have high variance and low bais. Bagging is not helpful in case of high bais.
