# SVM Classifier from scratch using python for Diabetes Prediction

## Overview
This project implements a Support Vector Machine (SVM) classifier from scratch using Python to perform linear classification on a diabetes dataset. The goal is to predict whether a patient has diabetes based on certain medical features. The implementation is done in a Google Colab notebook, and the code is available for easy access and execution.

## Table of Contents
- [Project Overview](#overview)
- [Dataset](#dataset)
- [Google Colab Notebook](#colab-notebook)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)

## Dataset
The diabetes dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset). It contains several medical diagnostic measurements and a target variable indicating the presence of diabetes in a patient.

### Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

The target variable is a binary label: 1 (diabetic) or 0 (non-diabetic).

## Google Colab Notebook
The code for this project is available in a Google Colab notebook which is mentioned in file above. You can access and run it directly without needing any local setup.

To run the notebook:
1. Open the link above.
2. Click `Run All` in Colab.
3. Follow the instructions within the notebook to train and test the model.

## Usage
All the code is included in the Colab notebook. You can make use of the notebook to run the SVM classifier and test the predictions. The notebook includes:
- Manual implementation of an SVM classifier for linear classification.
- Data loading and preprocessing.
- Model training and testing.

### Example:
The classifier can be used for predicting if a person has diabetes by inputting their medical features within the notebook.

```python
# Example usage in the notebook
input_features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
prediction = model.predict(input_features)
```

## Model
The SVM model was built from scratch within the Colab notebook using Python. Key aspects include:
- Formulating the SVM optimization problem.
- Implementing gradient descent for training.

## Results
The classifier achieved an accuracy of `77%` on the test set (you can replace this with your actual results).


## Challenges
- Implementing SVM from scratch in Python without libraries like Scikit-learn.
- Managing dataset imbalances or feature scaling.
- Ensuring convergence of the gradient descent method for linear classification.

## Future Improvements
- Add kernel methods for non-linear classification.
- Explore other optimization techniques for faster convergence.
- Extend the model to multi-class classification problems.
