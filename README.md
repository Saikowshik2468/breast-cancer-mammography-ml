# Breast Cancer Detection - Imbalanced Classification Project

A machine learning project for predicting breast cancer from severely imbalanced mammography data using the CLIMB-Mammography dataset.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed/cleaned data
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── data_processing/  # Data loading, preprocessing, and feature engineering
│   ├── models/          # Model definitions and training scripts
│   ├── utils/           # Utility functions and helpers
│   └── evaluation/      # Evaluation metrics and validation functions
├── streamlit_app/       # Streamlit dashboard application
├── models/              # Saved trained models
├── configs/             # Configuration files
├── tests/               # Unit tests
└── reports/             # Generated reports and visualizations
```

## Dataset

- **Name:** CLIMB-Mammography Dataset (Class-Imbalanced Learning Benchmark)
- **Size:** 11,183 samples with 2.32% positive cases (260 positive, 10,923 negative)
- **Class Imbalance Ratio:** 42:1
- **Features:** 6 continuous features (radius, texture, perimeter, area, smoothness, compactness)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the dataset:
   ```bash
   pip install imbalanced-ensemble
   ```

## Usage

### Data Exploration
Run the notebooks in the `notebooks/` directory for exploratory data analysis.

### Training Models
```bash
python src/models/train.py
```

### Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

## Evaluation Metrics

- F1-Score on minority class
- Balanced accuracy
- Precision-Recall AUC
- ROC-AUC
- Stratified cross-validation

## Techniques

- SMOTE/ADASYN for oversampling
- Cost-sensitive learning
- Threshold optimization
- Model persistence and pipeline serialization

