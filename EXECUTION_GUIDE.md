# Step-by-Step Execution Guide

## Prerequisites
- Python 3.8 or higher installed
- Terminal/Command Prompt access

---

## Step 1: Navigate to Project Directory

```powershell
cd "F:\ML Project - Breast cancer detect"
```

---

## Step 2: Create Virtual Environment (Recommended)

### Option A: Using venv
```powershell
python -m venv venv
```

### Option B: Using conda (if you have Anaconda/Miniconda)
```powershell
conda create -n breast_cancer_env python=3.10
conda activate breast_cancer_env
```

---

## Step 3: Activate Virtual Environment

### For venv (Windows PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

### For venv (Windows CMD):
```cmd
venv\Scripts\activate
```

### For conda:
```powershell
conda activate breast_cancer_env
```

---

## Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Expected output:** All packages will be installed including:
- numpy, pandas, scikit-learn
- imbalanced-learn, imbalanced-ensemble
- matplotlib, seaborn
- jupyter, streamlit
- etc.

---

## Step 5: Verify Installation

```powershell
python -c "import sklearn; import imblearn; import imbens; print('All packages installed successfully!')"
```

---

## Step 6: Execute Models

### Option A: Run Jupyter Notebooks (Recommended for Exploration)

#### 6.1. Start Jupyter Notebook Server
```powershell
jupyter notebook
```

This will open Jupyter in your browser. Then:

1. **Navigate to notebooks folder**
2. **Run notebooks in order:**
   - `01_EDA.ipynb` - Exploratory Data Analysis
   - `03_feature_eda.ipynb` - Feature Analysis
   - `04_train_test_split.ipynb` - Data Splitting
   - `05_baseline_model.ipynb` - Baseline Logistic Regression
   - `06_smote_model.ipynb` - SMOTE + Logistic Regression

#### 6.2. Or Run Specific Notebook via Command Line
```powershell
jupyter nbconvert --to notebook --execute notebooks/06_smote_model.ipynb --output executed_smote_model.ipynb
```

---

### Option B: Run Training Script (Python Script)

```powershell
python src/models/train.py
```

This will:
- Load the dataset
- Split into train/test
- Train a Random Forest model with SMOTE
- Evaluate and save the model

---

### Option C: Run Individual Analysis Scripts

#### Class Imbalance Analysis:
```powershell
python analyze_class_imbalance.py
```

#### Load Dataset Example:
```powershell
python load_dataset_example.py
```

---

## Step 7: Run Streamlit Dashboard (After Training)

```powershell
streamlit run streamlit_app/app.py
```

This will:
- Open a web browser with the dashboard
- Allow you to:
  - View data overview
  - Make predictions
  - See model performance

---

## Step 8: View Results

### Check Saved Models:
```powershell
dir models\
```

### Check Generated Reports:
```powershell
dir reports\
```

---

## Quick Execution Commands Summary

### Complete Workflow (Copy & Paste):
```powershell
# Navigate to project
cd "F:\ML Project - Breast cancer detect"

# Activate virtual environment (if using venv)
.\venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install -r requirements.txt

# Option 1: Run Jupyter Notebooks
jupyter notebook

# Option 2: Run training script
python src/models/train.py

# Option 3: Run Streamlit app
streamlit run streamlit_app/app.py
```

---

## Troubleshooting

### Issue: "jupyter: command not found"
**Solution:**
```powershell
pip install jupyter notebook
```

### Issue: "imblearn not found"
**Solution:**
```powershell
pip install imbalanced-learn
```

### Issue: "imbens not found"
**Solution:**
```powershell
pip install imbalanced-ensemble
```

### Issue: PowerShell execution policy error
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Virtual environment activation fails
**Solution:** Use full path:
```powershell
& "F:\ML Project - Breast cancer detect\venv\Scripts\Activate.ps1"
```

---

## Expected Execution Time

- **Data Loading**: ~5-10 seconds
- **EDA Notebooks**: ~30-60 seconds
- **Baseline Model Training**: ~10-30 seconds
- **SMOTE Model Training**: ~30-60 seconds (SMOTE takes time)
- **Streamlit App**: Instant (runs continuously)

---

## Next Steps After Execution

1. **Review Results**: Check confusion matrices and classification reports
2. **Compare Models**: Compare baseline vs SMOTE model performance
3. **Experiment**: Try different hyperparameters or algorithms
4. **Save Models**: Models are automatically saved to `models/` directory
5. **Generate Reports**: Export visualizations to `reports/` directory

---

## Notes

- All notebooks are independent and can be run in any order
- Models are saved automatically after training
- The Streamlit app requires a trained model to make predictions
- Use `random_state=42` for reproducibility

