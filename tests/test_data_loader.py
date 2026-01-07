"""
Tests for data loading functionality
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing.data_loader import load_mammography_data, load_data_as_dataframe


def test_load_mammography_data():
    """Test that data loads correctly"""
    X, y = load_mammography_data()
    
    assert X is not None
    assert y is not None
    assert len(X) == len(y)
    assert X.shape[1] == 6  # 6 features
    assert len(np.unique(y)) == 2  # Binary classification


def test_load_data_as_dataframe():
    """Test DataFrame loading"""
    df = load_data_as_dataframe()
    
    assert df is not None
    assert 'target' in df.columns
    assert len(df.columns) == 7  # 6 features + target
    assert df.shape[0] > 0

