# Diabetes Prediction using Machine Learning

This project aims to predict the likelihood of diabetes in individuals using various machine learning models. The project involves extensive data preprocessing, exploratory data analysis, and implementation of multiple ensemble learning techniques to achieve high prediction accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Usage](#usage)
- [Models Used](#models-used)
- [Performance Evaluation](#performance-evaluation)
- [Libraries](#libraries)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict whether a person has diabetes based on several health indicators. The project achieves approximately 90% prediction accuracy using a combination of machine learning models. The implementation focuses on balancing the dataset, optimizing model parameters, and utilizing ensemble learning techniques to improve prediction performance.

## Features

- **Menu System**: Provides a user interface to enable or disable specific functionalities without modifying the code.
- **Parameter Tuning**: Option to use precomputed optimal parameters or manually adjust them during model training.
- **Exploratory Data Analysis**:
  - Detects missing values and duplicates.
  - Visualizes data distributions and correlations using histograms, box plots, scatter plots, and heat maps.
- **Data Cleaning**:
  - Handles duplicates, anomalous values, and class imbalance using techniques like undersampling and normalization.
- **Machine Learning Models**: Implements various models including Stacking, Bagging, Boosting, KNN, Decision Trees, and Logistic Regression.



## Usage

1. Prepare your dataset in CSV format with the necessary features.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Follow the menu prompts to perform data analysis, train models, and evaluate results.

## Models Used

The project implements several machine learning models, with a focus on ensemble learning techniques:

1. **Stacking**:
   - Combines outputs from Random Forest, KNN, and Linear SVC using Logistic Regression.
2. **Bagging**:
   - Utilizes Decision Tree Classifiers with majority voting for improved performance.
3. **Boosting**:
   - Sequentially trains weak learners to correct errors made by previous models.
4. **K-Nearest Neighbors (KNN)**:
   - Evaluated using the elbow method for optimal neighbor selection.
5. **Decision Trees**:
   - Achieves 87% accuracy after hyperparameter tuning.
6. **Logistic Regression**:
   - A simple yet effective baseline model.

## Performance Evaluation

The models are evaluated using various metrics to ensure robust performance:

- **Cross-Validation**: Uses 5-fold cross-validation to estimate model generalizability.
- **F1 Score**: Balances precision and recall to provide a single metric of performance.
- **Recall**: Measures the ability of the model to correctly identify positive cases.
- **Accuracy**: Indicates the proportion of correctly predicted instances.
- **Log Loss**: Penalizes incorrect predictions based on the predicted probability.
- **Jaccard Index**: Measures similarity between predicted and actual labels.
- **Confusion Matrix**: Provides detailed insights into model classification errors.
- **ROC Curve and AUC**: Analyzes model performance across different thresholds, with AUC as an aggregate measure of the model's ability to distinguish between classes.

## Libraries

The project uses the following Python libraries:

- **Scikit-learn**: For machine learning models and evaluation metrics.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.

Install these libraries via:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

