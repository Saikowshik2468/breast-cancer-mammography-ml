"""
Data preprocessing pipeline for imbalanced classification
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into train and test sets with stratification
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify by target class
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling to balance the dataset
    
    Args:
        X_train: Training features
        y_train: Training targets
        random_state: Random seed
    
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled


def apply_adasyn(X_train, y_train, random_state=42):
    """
    Apply ADASYN oversampling to balance the dataset
    
    Args:
        X_train: Training features
        y_train: Training targets
        random_state: Random seed
    
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled


def create_preprocessing_pipeline(scaler=True, resampler=None):
    """
    Create a preprocessing pipeline
    
    Args:
        scaler: Whether to include scaling
        resampler: Resampling method ('smote', 'adasyn', or None)
    
    Returns:
        ImbPipeline: Preprocessing pipeline
    """
    steps = []
    
    if scaler:
        steps.append(('scaler', StandardScaler()))
    
    if resampler == 'smote':
        steps.append(('smote', SMOTE(random_state=42)))
    elif resampler == 'adasyn':
        steps.append(('adasyn', ADASYN(random_state=42)))
    
    if steps:
        return ImbPipeline(steps)
    else:
        return None

