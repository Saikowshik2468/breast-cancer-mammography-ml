"""
Class imbalance analysis and visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_class_distribution(y):
    """
    Analyze class distribution and return statistics
    
    Args:
        y: Target vector
    
    Returns:
        dict: Dictionary containing class distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    total_samples = len(y)
    benign_count = class_counts.get(0, 0)
    malignant_count = class_counts.get(1, 0)
    
    benign_percentage = (benign_count / total_samples) * 100
    malignant_percentage = (malignant_count / total_samples) * 100
    imbalance_ratio = benign_count / malignant_count if malignant_count > 0 else 0
    
    stats = {
        'total_samples': total_samples,
        'benign_count': benign_count,
        'malignant_count': malignant_count,
        'benign_percentage': benign_percentage,
        'malignant_percentage': malignant_percentage,
        'imbalance_ratio': imbalance_ratio
    }
    
    return stats


def print_class_distribution(y):
    """
    Print class distribution statistics
    
    Args:
        y: Target vector
    """
    stats = analyze_class_distribution(y)
    
    print("=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal samples: {stats['total_samples']:,}")
    print(f"\nClass Distribution:")
    print(f"  Benign (Class 0):     {stats['benign_count']:,} samples ({stats['benign_percentage']:.2f}%)")
    print(f"  Malignant (Class 1):  {stats['malignant_count']:,} samples ({stats['malignant_percentage']:.2f}%)")
    print(f"\nClass Imbalance Ratio: {stats['imbalance_ratio']:.1f}:1 (Benign:Malignant)")
    print("=" * 60)


def plot_class_distribution(y, title='Class Distribution in Mammography Dataset', 
                           save_path=None, show_plot=True):
    """
    Create a bar plot showing class distribution using seaborn
    
    Args:
        y: Target vector
        title: Plot title
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    stats = analyze_class_distribution(y)
    
    # Create DataFrame for visualization
    df_plot = pd.DataFrame({
        'Class': ['Benign (0)', 'Malignant (1)'],
        'Count': [stats['benign_count'], stats['malignant_count']],
        'Percentage': [stats['benign_percentage'], stats['malignant_percentage']]
    })
    
    # Create bar plot using seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_plot,
        x='Class',
        y='Count',
        palette=['#3498db', '#e74c3c'],  # Blue for benign, Red for malignant
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on top of bars
    for i, (count, pct) in enumerate(zip(df_plot['Count'], df_plot['Percentage'])):
        ax.text(i, count + max(df_plot['Count']) * 0.01, 
                f'{count:,}\n({pct:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Display plot
    if show_plot:
        plt.show()
    
    return plt.gcf()


def analyze_and_visualize_imbalance(y, title='Class Distribution in Mammography Dataset'):
    """
    Complete analysis: print statistics and create visualization
    
    Args:
        y: Target vector
        title: Plot title
    
    Returns:
        tuple: (stats_dict, figure)
    """
    # Print statistics
    print_class_distribution(y)
    
    # Create visualization
    fig = plot_class_distribution(y, title=title)
    
    # Get stats
    stats = analyze_class_distribution(y)
    
    return stats, fig

