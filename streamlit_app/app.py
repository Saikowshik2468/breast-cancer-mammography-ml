"""
Streamlit dashboard for Breast Cancer Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing.data_loader import load_data_as_dataframe
from evaluation.metrics import evaluate_model, print_classification_report


# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Breast Cancer Detection Dashboard")
st.markdown("Predicting breast cancer from imbalanced mammography data")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Model Prediction", "Model Performance"])

# Load data
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    return load_data_as_dataframe()

df = load_data()

if page == "Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Positive Cases", df['target'].sum())
    with col3:
        st.metric("Class Imbalance Ratio", f"{len(df[df['target']==0]) / df['target'].sum():.1f}:1")
    
    st.subheader("Class Distribution")
    st.bar_chart(df['target'].value_counts())
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Feature Statistics")
    st.dataframe(df.describe())

elif page == "Model Prediction":
    st.header("Make a Prediction")
    
    st.markdown("Enter mammography feature values to predict breast cancer:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.number_input("Radius", min_value=0.0, value=10.0, step=0.1)
        texture = st.number_input("Texture", min_value=0.0, value=15.0, step=0.1)
        perimeter = st.number_input("Perimeter", min_value=0.0, value=60.0, step=0.1)
    
    with col2:
        area = st.number_input("Area", min_value=0.0, value=300.0, step=1.0)
        smoothness = st.number_input("Smoothness", min_value=0.0, value=0.1, step=0.01)
        compactness = st.number_input("Compactness", min_value=0.0, value=0.2, step=0.01)
    
    # Load model if available
    models_dir = Path(__file__).parent.parent / 'models'
    model_path = models_dir / 'model.pkl'
    scaler_path = models_dir / 'scaler.pkl'
    
    if model_path.exists() and scaler_path.exists():
        @st.cache_resource
    def train_model():
    from src.models.train import train_and_save_model
    return train_and_save_model()

    model = train_model()

        scaler = joblib.load(scaler_path)
        
        if st.button("Predict"):
            # Prepare input
            input_data = np.array([[radius, texture, perimeter, area, smoothness, compactness]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ **Malignant (Cancer Detected)**")
                st.warning(f"Probability: {probability[1]:.2%}")
            else:
                st.success(f"✅ **Benign (No Cancer)**")
                st.info(f"Probability: {probability[0]:.2%}")
            
            # Show probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Benign', 'Malignant'],
                'Probability': probability
            })
            st.bar_chart(prob_df.set_index('Class'))
    else:
        st.warning("⚠️ Model not found. Please train a model first using `python src/models/train.py`")

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    models_dir = Path(__file__).parent.parent / 'models'
    model_path = models_dir / 'model.pkl'
    scaler_path = models_dir / 'scaler.pkl'
    
    if model_path.exists() and scaler_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Split data for evaluation
        from data_processing.preprocessing import split_data, scale_features
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        _, X_test_scaled, _ = scale_features(X_train, X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
        with col3:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        if 'roc_auc' in metrics:
            col5, col6 = st.columns(2)
            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            with col6:
                st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    else:
        st.warning("⚠️ Model not found. Please train a model first using `python src/models/train.py`")

