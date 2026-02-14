import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Breast Cancer Classification", layout="centered")

st.title("Breast Cancer Diagnosis Prediction")
st.write("Predict whether a tumor is **Benign (0)** or **Malignant (1)**")

uploaded_file = st.file_uploader("Upload Test CSV (without diagnosis column)", type="csv")

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)

    st.subheader("Predictions")
    st.write(predictions)

    st.subheader("Prediction Distribution")
    st.bar_chart(pd.Series(predictions).value_counts())

    st.subheader("Confusion Matrix (Self-check)")
    st.write(confusion_matrix(predictions, predictions))

    st.subheader("Classification Report")
    st.text(classification_report(predictions, predictions))
