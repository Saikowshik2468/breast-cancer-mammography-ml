# How to Run the Model - Quick Guide

## 🚀 Quick Start (Easiest Method)

### Option 1: Run the SMOTE Model Script
```powershell
python run_smote_model.py
```
**This will:**
- Load the dataset
- Train the SMOTE + Logistic Regression model
- Display all evaluation metrics
- Show confusion matrix and classification report

---

## 📓 Method 2: Run Jupyter Notebook (Interactive)

### Step 1: Start Jupyter Notebook
```powershell
jupyter notebook
```

### Step 2: Open the SMOTE Model Notebook
- Navigate to `notebooks/06_smote_model.ipynb`
- Click on it to open
- Run all cells (Cell → Run All) or run cells one by one (Shift + Enter)

**Advantages:**
- See visualizations (confusion matrix plots)
- Modify code interactively
- Step-by-step execution

---

## 🎯 Method 3: Run Training Script

```powershell
python src/models/train.py
```

**This will:**
- Train a Random Forest model with SMOTE
- Save the model to `models/` directory
- Display evaluation metrics

---

## 🌐 Method 4: Run Streamlit Dashboard (After Training)

```powershell
streamlit run streamlit_app/app.py
```

**This will:**
- Open a web browser with interactive dashboard
- Allow you to:
  - View data overview
  - Make predictions on new data
  - See model performance metrics

**Note:** Requires a trained model saved in `models/` directory

---

## 📊 Method 5: Run Individual Analysis Scripts

### Class Imbalance Analysis:
```powershell
python analyze_class_imbalance.py
```

### Load Dataset Example:
```powershell
python load_dataset_example.py
```

---

## 🔧 Prerequisites

### 1. Install Dependencies (First Time Only)
```powershell
pip install -r requirements.txt
```

### 2. Activate Virtual Environment (If Using)
```powershell
# For venv
.\venv\Scripts\Activate.ps1

# For conda
conda activate breast_cancer_env
```

---

## 📝 Complete Workflow Example

```powershell
# 1. Navigate to project directory
cd "F:\ML Project - Breast cancer detect"

# 2. Activate virtual environment (if using)
.\venv\Scripts\Activate.ps1

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Run the model
python run_smote_model.py
```

---

## 🎯 Recommended Execution Order

### For First-Time Users:
1. **Run EDA Notebooks** (to understand the data):
   ```powershell
   jupyter notebook notebooks/01_EDA.ipynb
   ```

2. **Run Feature Analysis**:
   ```powershell
   jupyter notebook notebooks/03_feature_eda.ipynb
   ```

3. **Run SMOTE Model**:
   ```powershell
   python run_smote_model.py
   ```
   OR
   ```powershell
   jupyter notebook notebooks/06_smote_model.ipynb
   ```

4. **Run Streamlit Dashboard** (after training):
   ```powershell
   streamlit run streamlit_app/app.py
   ```

---

## ⚡ Quick Commands Reference

| Task | Command |
|------|---------|
| Run SMOTE Model | `python run_smote_model.py` |
| Start Jupyter | `jupyter notebook` |
| Run Training Script | `python src/models/train.py` |
| Start Streamlit | `streamlit run streamlit_app/app.py` |
| Class Analysis | `python analyze_class_imbalance.py` |

---

## 🐛 Troubleshooting

### Issue: "python: command not found"
**Solution:** Use `python3` instead of `python`, or ensure Python is in PATH

### Issue: "ModuleNotFoundError"
**Solution:** 
```powershell
pip install -r requirements.txt
```

### Issue: "jupyter: command not found"
**Solution:**
```powershell
pip install jupyter notebook
```

### Issue: PowerShell execution policy error
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📈 Expected Output

When you run `python run_smote_model.py`, you should see:

1. ✅ Dataset loading confirmation
2. ✅ Training set/test set split information
3. ✅ Pipeline creation confirmation
4. ✅ Model training progress
5. ✅ SMOTE resampling statistics
6. ✅ **Evaluation Results:**
   - Accuracy score
   - Confusion Matrix
   - Malignant Class Metrics (Recall, Precision, F1-Score)
   - Complete Classification Report

---

## 💡 Tips

- **For Quick Testing:** Use `python run_smote_model.py`
- **For Exploration:** Use Jupyter Notebooks
- **For Presentation:** Use Streamlit Dashboard
- **For Production:** Use `src/models/train.py` and save models

---

## 📞 Need Help?

Check these files:
- `EXECUTION_GUIDE.md` - Detailed execution guide
- `QUICKSTART.md` - Quick start guide
- `README.md` - Project overview

