"""
Streamlit dashboard for Breast Cancer Detection
(Cloud-deployment safe version)
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_processing.data_loader import load_data_as_dataframe
from src.models.train import train_and_return_model
from src.evaluation.metrics import evaluate_model

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Detection Dashboard")
st.markdown("Predicting breast cancer from imbalanced mammography data")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Overview", "Model Prediction", "Model Performance"]
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return load_data_as_dataframe()

df = load_data()

# --------------------------------------------------
# Train model dynamically (ONCE)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model, scaler = train_and_return_model()
    return model, scaler

model, scaler = load_model()

# --------------------------------------------------
# DATA OVERVIEW PAGE
# --------------------------------------------------
if page == "Data Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Cancer Cases", int(df['target'].sum()))
    with col3:
        ratio = len(df[df['target'] == 0]) / df['target'].sum()
        st.metric("Imbalance Ratio", f"{ratio:.1f}:1")

    st.subheader("Class Distribution")
    st.bar_chart(df['target'].value_counts())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Feature Statistics")
    st.dataframe(df.describe())

# --------------------------------------------------
# MODEL PREDICTION PAGE
# --------------------------------------------------
elif page == "Model Prediction":
    st.header("Make a Prediction")

    st.markdown("Enter mammography feature values:")

    col1, col2 = st.columns(2)

    with col1:
        radius = st.number_input("Radius", 0.0, 50.0, 10.0, 0.1)
        texture = st.number_input("Texture", 0.0, 50.0, 15.0, 0.1)
        perimeter = st.number_input("Perimeter", 0.0, 200.0, 60.0, 0.1)

    with col2:
        area = st.number_input("Area", 0.0, 2000.0, 300.0, 1.0)
        smoothness = st.number_input("Smoothness", 0.0, 1.0, 0.1, 0.01)
        compactness = st.number_input("Compactness", 0.0, 1.0, 0.2, 0.01)

    if st.button("Predict"):
        input_data = np.array([[radius, texture, perimeter, area, smoothness, compactness]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Malignant (Cancer Detected)")
            st.warning(f"Probability: {probability[1]:.2%}")
        else:
            st.success("✅ Benign (No Cancer)")
            st.info(f"Probability: {probability[0]:.2%}")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Class": ["Benign", "Malignant"],
            "Probability": probability
        })
        st.bar_chart(prob_df.set_index("Class"))

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance Metrics")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    from data_processing.preprocessing import split_data, scale_features

    X_train, X_test, y_train, y_test = split_data(X, y)
    _, X_test_scaled, _ = scale_features(X_train, X_test)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
    col3.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    col4.metric("Recall", f"{metrics['recall']:.4f}")

    if "roc_auc" in metrics:
        col5, col6 = st.columns(2)
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        col6.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
