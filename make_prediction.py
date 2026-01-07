"""
Quick script to make predictions using the trained SMOTE model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from data_processing.data_loader import load_mammography_data

# Load data and train model
print("="*70)
print("LOADING MODEL...")
print("="*70)
X, y = load_mammography_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
print("[OK] Model trained and ready!")

# ============================================================================
# ENTER PATIENT DATA HERE
# ============================================================================
# Format: [radius, texture, perimeter, area, smoothness, compactness]
# Example values:
new_patient = np.array([[10.0, 15.0, 60.0, 300.0, 0.10, 0.20]])

# Make prediction
prediction = pipeline.predict(new_patient)[0]
probability = pipeline.predict_proba(new_patient)[0]

# Display results
print("\n" + "="*70)
print("PREDICTION RESULT")
print("="*70)
print(f"\nPatient Features:")
print(f"  Radius:      {new_patient[0][0]:.2f}")
print(f"  Texture:     {new_patient[0][1]:.2f}")
print(f"  Perimeter:   {new_patient[0][2]:.2f}")
print(f"  Area:        {new_patient[0][3]:.2f}")
print(f"  Smoothness:  {new_patient[0][4]:.2f}")
print(f"  Compactness: {new_patient[0][5]:.2f}")

print(f"\n{'='*70}")
if prediction == 1:
    print("⚠️  PREDICTION: MALIGNANT (Cancer Detected)")
    print(f"   Risk Level: {probability[1]*100:.2f}%")
else:
    print("✅ PREDICTION: BENIGN (No Cancer)")
    print(f"   Confidence: {probability[0]*100:.2f}%")

print(f"\nDetailed Probabilities:")
print(f"  Benign:    {probability[0]:.4f} ({probability[0]*100:.2f}%)")
print(f"  Malignant: {probability[1]:.4f} ({probability[1]*100:.2f}%)")
print("="*70)

print("\n💡 Tip: Edit 'new_patient' values in this script to test different patients!")

