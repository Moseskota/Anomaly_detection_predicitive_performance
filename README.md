# Anomaly_detection_predicitive_performance

This repository contains a colab Notebook for Anomaly detection by employing machine learning techniques especially to handle imbalanced datasets. The notebook demonstrates the use of SMOTE (Synthetic Minority Over-sampling Technique) for oversampling and RandomUnderSampler for undersampling to balance the dataset.

**Overview**
The dataset used in this notebook is highly imbalanced, with the majority class representing normal conditions and the minority class representing anomalous conditions. To effectively train a machine learning model on this imbalanced dataset, the following techniques are applied:

SMOTE (Synthetic Minority Over-sampling Technique): This technique is used to generate synthetic samples for the minority class to balance the dataset.
RandomUnderSampler: This technique randomly removes samples from the majority class to balance the dataset.

Notebook Contents
The notebook includes the following sections:

**Data Loading and Exploration**:
Load and explore the dataset.
Understand the class distribution.
Link for the dataset -- https://www.kaggle.com/datasets/mosescalvin/anomaly-data 

**Model selection**
selecting the best model with cross validation
performing over and under sampling techniques to obtain the best accuracy

**Oversampling with SMOTE**:
Apply SMOTE to balance the dataset.
Train a Logistic Regression model on the oversampled data.
Evaluate the model using a confusion matrix and classification report.
Undersampling with RandomUnderSampler:

**Apply RandomUnderSampler to balance the dataset**.
Train a Logistic Regression model on the undersampled data.
Evaluate the model using a confusion matrix and classification report.

**Results Visualization**:
Visualize the class distribution before and after sampling.
Plot confusion matrices for model evaluations.
