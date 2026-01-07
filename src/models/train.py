"""
Model training script for imbalanced classification
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.data_loader import load_mammography_data
from data_processing.preprocessing import split_data, scale_features, apply_smote
from evaluation.metrics import evaluate_model, print_classification_report


def train_model(X_train, y_train, model_type='random_forest', use_smote=True):
    """
    Train a classification model
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model ('random_forest', 'logistic', 'svm')
        use_smote: Whether to apply SMOTE oversampling
    
    Returns:
        Trained model
    """
    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    elif model_type == 'svm':
        model = SVC(
            class_weight='balanced',
            random_state=42,
            probability=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def main():
    """Main training function"""
    # Load data
    print("Loading data...")
    X, y = load_mammography_data()
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    print("Training model...")
    model = train_model(X_train_scaled, y_train, model_type='random_forest', use_smote=True)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred)
    print_classification_report(y_test, y_pred)
    
    # Save model and scaler
    models_dir = Path(__file__).parent.parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / 'model.pkl')
    joblib.dump(scaler, models_dir / 'scaler.pkl')
    print(f"\nModel saved to {models_dir / 'model.pkl'}")
    print(f"Scaler saved to {models_dir / 'scaler.pkl'}")


if __name__ == "__main__":
    main()

