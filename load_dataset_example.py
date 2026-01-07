"""
Example: Load mammography dataset using imblearn.datasets.fetch_datasets
"""

import numpy as np
from imblearn.datasets import fetch_datasets

# Load the mammography dataset
data = fetch_datasets('mammography')

# Split into features X and target y
X = data.data
y = data.target

# Display basic information
print(f"Dataset loaded successfully!")
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"\nClass distribution:")
print(f"  Class 0 (Benign): {np.sum(y == 0)}")
print(f"  Class 1 (Malignant): {np.sum(y == 1)}")
print(f"  Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.1f}:1")

# Display first few samples
print(f"\nFirst 5 feature vectors (X):")
print(X[:5])
print(f"\nFirst 5 target values (y):")
print(y[:5])

