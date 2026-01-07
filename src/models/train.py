"""
Model training module for imbalanced classification
(Streamlit-compatible version)
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.data_loader import load_mammography_data
from data_processing.preprocessing import split_data, scale_features, apply_smote


def train_model(X_train, y_train, model_type='random_forest', use_smote=True):
    """
    Train a classification model
    """
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

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
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def train_and_return_model(model_type='random_forest'):
    """
    Streamlit-safe training function.
    Trains model dynamically and returns model + scaler.
    """
    # Load data
    X, y = load_mammography_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    model = train_model(
        X_train_scaled,
        y_train,
        model_type=model_type,
        use_smote=True
    )

    return model, scaler


# Optional: keep local training capability
if __name__ == "__main__":
    model, scaler = train_and_return_model()

    models_dir = Path(__file__).parent.parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / 'model.pkl')
    joblib.dump(scaler, models_dir / 'scaler.pkl')

    print("Model and scaler saved locally.")
