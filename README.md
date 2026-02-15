🩺 Breast Cancer Classification – Machine Learning & Streamlit App
1. Problem Statement

The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict whether a breast tumor is Malignant or Benign, based on numerical features extracted from digitized images of fine needle aspirate (FNA) of breast masses.
An interactive Streamlit web application is developed and deployed to demonstrate model predictions, evaluation metrics, and comparisons.

2. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Number of Instances: 569

Number of Features: 30 numeric features

Target Variable: diagnosis

M → Malignant (1)

B → Benign (0)

Dataset Characteristics

Fully numerical dataset

No missing values in original data

Well-balanced for binary classification

Suitable for both classical and ensemble ML models

3. Models Used and Evaluation Metrics

All models were trained and evaluated on the same dataset using a consistent preprocessing pipeline (feature scaling and stratified train–test split).

Machine Learning Models Implemented

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

Evaluation Metrics

The following metrics were calculated for each model:

Accuracy

Area Under the ROC Curve (AUC)

Precision

Recall

F1-score

Matthews Correlation Coefficient (MCC)

4. Model Performance Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1-score	MCC
Logistic Regression	0.9650	0.9962	0.98	0.9245	0.9515	0.9251
Decision Tree	0.9580	0.9473	0.98	0.9057	0.9412	0.9103
KNN	0.9580	0.9860	0.98	0.9057	0.9412	0.9103
Naive Bayes	0.9441	0.9925	0.96	0.8868	0.9216	0.8798
Random Forest	0.9580	0.9960	1.00	0.8868	0.9400	0.9120
XGBoost	0.9720	0.9937	1.00	0.9245	0.9608	0.9408
5. Observations on Model Performance
ML Model	Observation
Logistic Regression	Strong baseline model with excellent AUC and balanced performance
Decision Tree	Performs well but slightly prone to overfitting
KNN	Competitive performance, sensitive to feature scaling
Naive Bayes	Fast and simple, but assumptions limit recall
Random Forest	Very high precision and strong overall robustness
XGBoost	Best overall model with highest accuracy, F1-score, and MCC
6. Streamlit Web Application

The Streamlit app provides an interactive interface to:

Upload a test CSV file

Select a machine learning model

View predictions and probabilities

Display evaluation metrics

Compare all model performances in a single table

Visualize a confusion matrix

Note: Metrics shown in the Streamlit app are for demonstration purposes using uploaded test data.
The true evaluation metrics are computed offline during model training.

7. Project Structure
ML_Assignment_2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- breast_cancer.csv
│-- model/
│   ├── Logistic Regression.pkl
│   ├── Decision Tree.pkl
│   ├── KNN.pkl
│   ├── Naive Bayes.pkl
│   ├── Random Forest.pkl
│   ├── XGBoost.pkl
│   ├── scaler.pkl
│   └── metrics.csv

8. How to Run Locally
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python model/train_models.py
python -m streamlit run app.py

9. Deployment

The application is deployed using Streamlit Community Cloud.
A public link to the live application is provided as part of the assignment submission.

10. Conclusion

This project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, model training, evaluation, interactive visualization, and deployment.
Ensemble models, particularly XGBoost, achieved the best performance, highlighting their effectiveness for structured medical datasets.
