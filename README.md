# Breast-Cancer-Prediction
This project aims to predict whether a tumor is benign or malignant based on cell sample features using machine learning.
---

## Task Description

- The goal of this project is to build a machine learning model that can classify breast tumors as either benign or malignant based on a set of medical attributes obtained from fine needle aspirate (FNA) of breast masses.
- Dataset:  Breast Cancer Wisconsin (Diagnostic) Dataset
- Target: Tumor Class (Benign = 0, Malignant = 1)

-----

## Tools & Libraries

- pandas
- numpy
- scikit-learn
- gradio
- matplotlib
- seaborn

----

## Topics Covered

# Machine Learning

- Supervised Learning
- Binary Classification
- Model Training and Evaluation
- Hyperparameter Tuning (RandomizedSearchCV)
- Model Comparison: Support Vector Machine vs Random Forest

# Data Preprocessing & Analysis
#  Model Evaluation
# Deployment & UI

---

## Steps Performed
This section outlines the complete end-to-end process followed in the Breast Cancer Prediction project:

# 1. Problem Definition
- Objective: Predict whether a breast tumor is benign or malignant.
- Type: Binary classification using supervised learning.

# 2. Dataset Collection & Preparation
- Loaded the Breast Cancer Wisconsin Dataset.
- Handled missing values in the Bare_Nuclei column.
- Dropped unnecessary columns (like ID).
- Converted labels (2 = Benign → 0, 4 = Malignant → 1).

# 3. Exploratory Data Analysis (EDA)
- Visualized feature distributions.
- Plotted feature correlation heatmap.
- Checked class distribution to ensure balance.

# 4. Feature Scaling
- Applied StandardScaler to normalize features before training models.

# 5. Train-Test Split
- Split the data into 80% training and 20% testing using train_test_split.

# 6. Model Selection & Training
- Trained Support Vector Machine (SVM) and Random Forest Classifier.
- Used RandomizedSearchCV to tune hyperparameters efficiently.

# 7. Model Evaluation
-Evaluated models on test set using:
- Accuracy
- Precision
- Recall
- Classification Report
- Confusion Matrix

# 8. Model Deployment with Gradio
- Created an interactive Gradio web app.
- Included slider inputs for 9 cell-level medical features.
- Returned real-time predictions:
- 🟢 Benign
- 🔴 Malignant

  ##  Results
  The Breast Cancer Prediction models were trained, tuned, and evaluated using standard metrics. Below are the performance results for both Support Vector Machine (SVM) and Random Forest Classifier after hyperparameter tuning with RandomizedSearchCV .

  
