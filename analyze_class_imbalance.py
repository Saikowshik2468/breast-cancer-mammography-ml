"""
Analyze and visualize class imbalance in the mammography dataset

This script demonstrates how to analyze and visualize class imbalance
using the reusable functions from the data_processing module.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.data_loader import load_mammography_data
from data_processing.class_imbalance import analyze_and_visualize_imbalance

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def main():
    """Main function to load data and analyze class imbalance"""
    print("Loading mammography dataset...")
    X, y = load_mammography_data()
    
    print(f"\nDataset loaded: {X.shape[0]:,} samples, {X.shape[1]} features\n")
    
    # Analyze and visualize class imbalance
    stats, fig = analyze_and_visualize_imbalance(
        y, 
        title='Class Distribution in Mammography Dataset'
    )
    
    return stats, fig


if __name__ == "__main__":
    stats, fig = main()

