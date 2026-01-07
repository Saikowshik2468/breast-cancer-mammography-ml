"""
Visualize the SMOTE model - Show model details, performance metrics, and visualizations
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    recall_score,
    f1_score,
    precision_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_processing.data_loader import load_mammography_data

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 70)
print("MODEL VISUALIZATION AND INSPECTION")
print("=" * 70)

# Load and prepare data
print("\n[1/6] Loading dataset...")
X, y = load_mammography_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train model
print("[2/6] Creating and training model...")
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print("[3/6] Model trained successfully!")

# 1. Show Model Structure
print("\n" + "=" * 70)
print("MODEL STRUCTURE")
print("=" * 70)
print("\nPipeline Steps:")
for i, (step_name, step) in enumerate(pipeline.steps, 1):
    print(f"  {i}. {step_name}: {type(step).__name__}")
    if hasattr(step, 'get_params'):
        params = step.get_params()
        if step_name == 'classifier':
            print(f"     - C: {params.get('C', 'default')}")
            print(f"     - max_iter: {params.get('max_iter', 'default')}")
            print(f"     - class_weight: {params.get('class_weight', 'default')}")
        elif step_name == 'smote':
            print(f"     - k_neighbors: {params.get('k_neighbors', 'default')}")
            print(f"     - random_state: {params.get('random_state', 'default')}")

# 2. Show Model Coefficients (Logistic Regression)
print("\n" + "=" * 70)
print("MODEL COEFFICIENTS (Logistic Regression)")
print("=" * 70)
feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness']
classifier = pipeline.named_steps['classifier']
coefficients = classifier.coef_[0]
intercept = classifier.intercept_[0]

print("\nFeature Coefficients:")
print("-" * 70)
for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
    print(f"{feature:15s}: {coef:10.4f} {'(positive impact)' if coef > 0 else '(negative impact)'}")
print("-" * 70)
print(f"{'Intercept':15s}: {intercept:10.4f}")
print("\nInterpretation:")
print("- Positive coefficients increase probability of malignant class")
print("- Negative coefficients decrease probability of malignant class")
print("- Larger absolute values indicate stronger influence")

# 3. Confusion Matrix Visualization
print("\n[4/6] Creating confusion matrix visualization...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot 1: Confusion Matrix (Counts)
ax1 = fig.add_subplot(gs[0, 0])
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp1.plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=10)

# Plot 2: Confusion Matrix (Percentages)
ax2 = fig.add_subplot(gs[0, 1])
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=['Benign', 'Malignant'])
disp2.plot(ax=ax2, cmap='Blues', values_format='.2f')
ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold', pad=10)

# Plot 3: ROC Curve
print("[5/6] Creating ROC curve...")
ax3 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=10)
ax3.legend(loc="lower right")
ax3.grid(alpha=0.3)

# Plot 4: Precision-Recall Curve
print("[6/6] Creating Precision-Recall curve...")
ax4 = fig.add_subplot(gs[1, 1])
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
ax4.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
baseline = np.sum(y_test == 1) / len(y_test)
ax4.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=10)
ax4.legend(loc="lower left")
ax4.grid(alpha=0.3)

# Plot 5: Feature Importance (Coefficients)
ax5 = fig.add_subplot(gs[2, :])
colors = ['red' if c < 0 else 'green' for c in coefficients]
bars = ax5.barh(feature_names, coefficients, color=colors, edgecolor='black', linewidth=1.5)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax5.set_title('Feature Importance (Logistic Regression Coefficients)', fontsize=14, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3)
for i, (bar, coef) in enumerate(zip(bars, coefficients)):
    ax5.text(coef + (0.01 if coef >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
             f'{coef:.3f}', ha='left' if coef >= 0 else 'right', va='center', fontweight='bold')

plt.suptitle('SMOTE + Logistic Regression Model - Complete Visualization', 
             fontsize=16, fontweight='bold', y=0.995)

print("\nSaving visualization to 'model_visualization.png'...")
plt.savefig('model_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved!")
plt.show()

# 4. Print Performance Metrics
print("\n" + "=" * 70)
print("PERFORMANCE METRICS SUMMARY")
print("=" * 70)
malignant_recall = recall_score(y_test, y_pred, pos_label=1)
malignant_precision = precision_score(y_test, y_pred, pos_label=1)
malignant_f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\nMalignant Class (Class 1) Metrics:")
print(f"  Recall (Sensitivity):    {malignant_recall:.4f} ({malignant_recall*100:.2f}%)")
print(f"  Precision:               {malignant_precision:.4f} ({malignant_precision*100:.2f}%)")
print(f"  F1-Score:                {malignant_f1:.4f} ({malignant_f1*100:.2f}%)")
print(f"\nOverall Metrics:")
print(f"  ROC-AUC:                 {roc_auc:.4f}")
print(f"  PR-AUC:                  {pr_auc:.4f}")
print(f"  Accuracy:                {np.mean(y_pred == y_test):.4f} ({np.mean(y_pred == y_test)*100:.2f}%)")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nThe model visualization has been displayed and saved as 'model_visualization.png'")
print("\nYou can see:")
print("  1. Model structure and parameters")
print("  2. Feature coefficients (importance)")
print("  3. Confusion matrix (counts and percentages)")
print("  4. ROC curve (model discrimination ability)")
print("  5. Precision-Recall curve (imbalanced data performance)")
print("  6. Feature importance bar chart")

