# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the dataset library:**
   ```bash
   pip install imbalanced-ensemble
   ```

## Running the Project

### 1. Exploratory Data Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Train a Model
```bash
python src/models/train.py
```

This will:
- Load the CLIMB-Mammography dataset
- Split into train/test sets
- Apply feature scaling
- Train a Random Forest classifier with SMOTE
- Evaluate the model
- Save the model and scaler to `models/` directory

### 3. Run Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

The dashboard includes:
- **Data Overview**: Dataset statistics and visualizations
- **Model Prediction**: Interactive interface to make predictions
- **Model Performance**: Evaluation metrics and performance analysis

## Project Workflow

1. **Data Exploration** → `notebooks/01_EDA.ipynb`
2. **Data Processing** → `src/data_processing/`
3. **Model Training** → `src/models/train.py`
4. **Evaluation** → `src/evaluation/metrics.py`
5. **Visualization** → `streamlit_app/app.py`

## Configuration

Edit `configs/config.yaml` to modify:
- Data splitting parameters
- Preprocessing options (SMOTE, ADASYN)
- Model hyperparameters
- Evaluation metrics

## Testing

Run tests:
```bash
pytest tests/
```

