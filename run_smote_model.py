"""
Quick script to run SMOTE model and display results
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    f1_score,
    precision_score,
    accuracy_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_processing.data_loader import load_mammography_data

print("=" * 70)
print("SMOTE + LOGISTIC REGRESSION MODEL")
print("=" * 70)
print("\nStep 1: Loading dataset...")
X, y = load_mammography_data()
print(f"[OK] Dataset loaded: {X.shape[0]:,} samples, {X.shape[1]} features")

print("\nStep 2: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Training set: {X_train.shape[0]:,} samples")
print(f"[OK] Test set: {X_test.shape[0]:,} samples")

print("\nStep 3: Creating pipeline (StandardScaler -> SMOTE -> LogisticRegression)...")
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
print("[OK] Pipeline created")

print("\nStep 4: Training model with SMOTE oversampling...")
pipeline.fit(X_train, y_train)
print("[OK] Model trained successfully")

# Get resampled data info
X_resampled, y_resampled = pipeline.named_steps['smote'].fit_resample(
    pipeline.named_steps['scaler'].transform(X_train), y_train
)
print(f"[OK] After SMOTE: {len(y_resampled):,} samples (balanced classes)")

print("\nStep 5: Making predictions...")
y_pred = pipeline.predict(X_test)
print("[OK] Predictions generated")

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)
print(f"{'':<20s} {'Predicted Benign':<20s} {'Predicted Malignant':<20s}")
print("-" * 70)
print(f"{'Actual Benign':<20s} {cm[0, 0]:<20d} {cm[0, 1]:<20d}")
print(f"{'Actual Malignant':<20s} {cm[1, 0]:<20d} {cm[1, 1]:<20d}")
print("-" * 70)
print(f"\nTrue Negatives (TN):  {tn:>6d}")
print(f"False Positives (FP): {fp:>6d}")
print(f"False Negatives (FN): {fn:>6d}")
print(f"True Positives (TP):   {tp:>6d}")

# Malignant class metrics
malignant_recall = recall_score(y_test, y_pred, pos_label=1)
malignant_precision = precision_score(y_test, y_pred, pos_label=1)
malignant_f1 = f1_score(y_test, y_pred, pos_label=1)

print("\n" + "-" * 70)
print("MALIGNANT CLASS (Class 1) METRICS")
print("-" * 70)
print(f"Recall (Sensitivity): {malignant_recall:.4f} ({malignant_recall*100:.2f}%)")
print(f"Precision:            {malignant_precision:.4f} ({malignant_precision*100:.2f}%)")
print(f"F1-Score:             {malignant_f1:.4f} ({malignant_f1*100:.2f}%)")
print("-" * 70)

# Classification Report
print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
print("=" * 70)

print("\n[OK] Model evaluation completed!")

