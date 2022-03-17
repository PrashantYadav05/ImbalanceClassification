# Imbalanced Classification
Imbalanced classification is the problem of classification when there is an unequal distribution of classes in the training dataset.
## Causes of class imbalance:
Biased Sampling & Measurement Errors.
## Intuition for Imbalanced Classification:
* Create and plot a binary classification problem
* Create synthetic dataset with class Distribution
* Effect of skewed class distributions
## Challenge of Imbalanced Classification:
* Compounding effect of dataset size.
* Compounding effect of label noise (Label noise refers to examples that belong to one class that are labeled as another class)
* Compounding effect of data distribution.
## Evaluation Metrics:
* Challenge of choosing metrics for classification, and how it is particularly difficult when there is a skewed class distribution.
* How there are three main types of metrics for evaluating classifier models, referred to as rank, threshold, and probability.
* How to choose a metric for imbalanced classification if you don't know where to start.

|                   | Positive Prediction | Negative Prediction |
|-------------------|-------------------- |---------------------|
|**Positive Class** | True Positive (TP)  | False Negative (FN) |
|**Negative Class** | False Positive (FP) | True Negative (TN)  |

Senitivity: 

* Are you predicting probabilities?
  * Do you need class labels?
    * Is the positive class more important?
      * Use Precision-Recall AUC
    * Are both classes important?
      * Use ROC AUC
  * Do you need probabilities?
    * Use Brier Score and Brier Skill Score
* Are you predicting class labels?
  * Is the positive class more important?
    * Are False Negatives and False Positives Equally Important?
      * Use F1-measure
    * Are False Negatives More Important?
      * Use F2-measure
    * Are False Positives More Important?
      * Use F0.5-measure
  * Are both classes important?
    * Do you have < 80%-90% Examples for the Majority Class?
      * Use Accuracy
    * Do you have > 80%-90% Examples for the Majority Class?
      * Use G-mean
## Data Sampling:

* Random Sampling
* Random Oversampling: Randomly duplicate examples in the minority class.
* Random Undersampling: Randomly delete examples in the majority class.
### Oversampling 
* **Synthetic Minority Oversampling Technique (SMOTE):** A random example from the minority class is rst chosen. Then k of the nearest neighbors for that example are found (typically k = 5). A randomly selected neighbor is chosenand a synthetic example is created at a randomly selected point between the two examples in feature space. As described in the *Nitesh Chawla, et al. in their 2002 paper*, it suggests firstrst using random undersampling to trim the number of examples in the majority class, then use SMOTE to oversample the minority class to balance the class distribution.
**Limitation:** A general downside of the approach is that synthetic examples are created without consideringthe majority class, possibly resulting in ambiguous examples if there is a strong overlap for the classes.
* **Borderline-SMOTE:** A popular extension to SMOTE involves selecting those instances of the minority class that are misclassified, such as with a k-nearest neighbor classification model. We can then oversample just those difficult instances, providing more resolution only where it may be required.
* **Borderline-SMOTE SVM:** An alternative to Borderline-SMOTE where a SVM algorithm is used instead of a KNN to identify misclassified examples on the decision boundary.
* **Adaptive Synthetic Sampling (ADASYN):** Generating synthetic samples inversely proportional to the density of the examples in the minority class. That is, generate more synthetic examples in regions of the feature space where the density of minority examples is low, and fewer or none where the density is high.

### Undersampling
The simplest undersampling technique involves randomly selecting examples from the majority class and deleting them from the training dataset. This is referred to as randomundersampling. Although simple and effective, a limitation of this technique is that examplesare removed without any concern for how useful or important they might be in determiningthe decision boundary between the classes. This means it is possible, or even likely, that usefulinformation will be deleted.

*  Methods that Select Examples to Keep:
    * **Near Miss Undersampling:** Refers to a collection of undersampling methods that select examples based on the distance of majority class examples to minority class examples. **NearMiss-1:** Majority class examples with minimum average distance to three closest minority class examples. **NearMiss-2:** Majority class examples with minimum average distance to three furthest minority class examples. **NearMiss-3:** Majority class examples with minimum distance to each minority class example.
    * **Condensed Nearest Neighbor Rule Undersampling**
* Methods that Select Examples to Delete
    * **Tomek Links for Undersampling**
    * **Edited Nearest Neighbors Rule for Undersampling**
* Combinations of Keep and Delete Methods
    * **One-Sided Selection for Undersampling**
    * **Neighborhood Cleaning Rule for Undersampling**
## Oversampling and Undersampling
* Binary Test Problem and Decision Tree Model
* Manually Combine Data Sampling Method
    * **Random Oversampling and Undersampling**
    * **SMOTE and Random Undersampling**
* Standard Combined Data Sampling Methods
    * **SMOTE and Tomek Links Undersampling**
    * **SMOTE and Edited Nearest Neighbors Undersampling*** 

## Cost-Sensitive Learning
Classifying a majority class as minority is typically far less of a problem than classifying a minority class as a majority.
**Bank Loan Problem:** Denying a loan to a good customer is not as bad as giving a loan to a bad customer that may never repay it.
**Cancer Diagnosis Problem:** It is better to diagnose a healthy patient with cancer and follow-up with more medical tests than it is to discharge a patient that has cancer.
**Fraud Detection Problem:** Identifying good claims as fraudulent and following up with the customer is better than honoring
fraudulent insurance claims. 
It can see with these examples that misclassification errors are not desirable in general, but one type of misclassification is much worse than the other. Specifically predicting positive cases as a negative case is more harmful, more expensive, or worse in whatever way we want to measure the context of the target domain. As such, both the underrepresentation of the minority class in the training data and the increased importance on correctly identifying examples from the minority class make imbalanced classication one of the most challenging problems in applied machine learning.

A subfield of machine learning that is focused on learning and using models on data that have uneven penalties or costs when making predictions and more. This field is generally referred to as **Cost-Sensitive Machine Learning**, or more simply Cost-Sensitive Learning.
In cost-sensitive learning, a penalty associated with an incorrect prediction and is referred to as a cost. The goal of cost-sensitive learning is to minimize the cost of a model on the training dataset, where it is assumed that different types of prediction errors have a different and known associated cost.

The existing machine learning algorithms can be modified to make use of the cost matrix. This might involve a modification that is unique to each algorithm and which can be quite time consuming to develop and test. Many such algorithm-specific augmentations have been proposed for popular algorithms, like decision trees and support vector machines. The scikit-learn Python machine learning library provides examples of these cost-sensitive extensions via the class weight argument on the **SVC** & **DecisionTreeClassifier** classifiers. Another more general approach to modifying existing algorithms is to use the costs as a
penalty for misclassifition when the algorithms are trained. Given that most machine learning lgorithms are trained to minimize error, cost for misclassifition is added to the error or used o weigh the error during the training process.
This approach can be used for iteratively trained algorithms, such as logistic regression and rtifial neural networks. The scikit-learn library provides examples of these cost-sensitive xtensions via the class weight argument on the **LogisticRegression** & **RidgeRegression** classifiers. 


Not All Classification Errors Are Equal
* Cost-Sensitive Learning
* Cost-Sensitive Imbalanced Classification
* Cost-Sensitive Methods

## Probability Caliberation:

## One-Class Classification
## Project: Phoneme Classification
## project: Microcalcification Classification
