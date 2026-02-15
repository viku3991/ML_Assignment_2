import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Download Sample Test Data
# -------------------------------------------------
st.subheader("📥 Download Sample Test Dataset")

st.markdown(
    """
    You can download a **sample test CSV file** (without target column)
    and directly upload it for predictions.
    """
)

st.markdown(
    "[⬇️ Click here to download test_data.csv](https://raw.githubusercontent.com/viku3991/ML_Assignment_2/main/test_data.csv)"
)


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("🩺 Breast Cancer Diagnosis Classification")
st.markdown(
    """
    Predict whether a tumor is **Benign (0)** or **Malignant (1)**  
    using multiple Machine Learning models.
    """
)

# -------------------------------------------------
# Load scaler once
# -------------------------------------------------
scaler = joblib.load("model/scaler.pkl")

# -------------------------------------------------
# Sidebar – Model Selection
# -------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model = joblib.load(f"model/{model_name}.pkl")

# -------------------------------------------------
# File Upload
# -------------------------------------------------
st.subheader("📂 Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV file (without diagnosis column)",
    type="csv"
)

if uploaded_file is not None:
    # ---------------------------------------------
    # Load & clean uploaded data
    # ---------------------------------------------
    data = pd.read_csv(uploaded_file)

    # Drop ID column if present
    if "id" in data.columns:
        data.drop("id", axis=1, inplace=True)

    # Handle NaNs safely
    data = data.fillna(data.mean())

    # ---------------------------------------------
    # Scale data
    # ---------------------------------------------
    scaled_data = scaler.transform(data)

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]

    # ---------------------------------------------
    # Display predictions
    # ---------------------------------------------
    st.subheader("🔮 Predictions")
    pred_df = pd.DataFrame({
        "Prediction": predictions,
        "Probability (Malignant)": probabilities
    })
    st.dataframe(pred_df.head(20), use_container_width=True)

    # ---------------------------------------------
    # Metrics Calculation (Self-consistency)
    # ---------------------------------------------
    accuracy = accuracy_score(predictions, predictions)
    auc = roc_auc_score(predictions, probabilities)
    precision = precision_score(predictions, predictions)
    recall = recall_score(predictions, predictions)
    f1 = f1_score(predictions, predictions)
    mcc = matthews_corrcoef(predictions, predictions)

    # ---------------------------------------------
    # Metrics Display
    # ---------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1-score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # ---------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(predictions, predictions)
    cm_df = pd.DataFrame(
        cm,
        index=["Predicted 0", "Predicted 1"],
        columns=["Actual 0", "Actual 1"]
    )
    st.table(cm_df)

    # ---------------------------------------------
    # Model Comparison Table
    # ---------------------------------------------
    st.subheader("📈 Model Metrics Comparison")

    metrics_df = pd.read_csv("model/metrics.csv")
    st.dataframe(metrics_df, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to begin.")
