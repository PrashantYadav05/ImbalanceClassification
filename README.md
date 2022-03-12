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
* Compounding effect of label noise.
* Compounding effect of data distribution.
## Evaluation Metrics:
* Challenge of choosing metrics for classification, and how it is particularly difficult when there is a skewed class distribution.
* How there are three main types of metrics for evaluating classifier models, referred to as rank, threshold, and probability.
* How to choose a metric for imbalanced classification if you don't know where to start.

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


* Synthetic Minority Oversampling Technique
* SMOTE for Balancing Data
* SMOTE for Classification
* SMOTE With Selective Sample Generation

Undersampling for Imbalanced Classication
*  Methods that Select Examples to Keep
* Methods that Select Examples to Delete
* Combinations of Keep and Delete Methods

## Cost-Sensitive Learning

Not All Classification Errors Are Equal
* Cost-Sensitive Learning
* Cost-Sensitive Imbalanced Classification
* Cost-Sensitive Methods

## Probability Caliberation:

## One-Class Classification
## Project: Phoneme Classification
## project: Microcalcification Classification
