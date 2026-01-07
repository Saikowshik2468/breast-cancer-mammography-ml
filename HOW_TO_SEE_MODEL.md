# How to See/Visualize the Model

## 🎯 Multiple Ways to View Your Model

---

## Method 1: Run Visualization Script (Recommended)

### Quick Visualization with Plots
```powershell
python visualize_model.py
```

**This will show:**
- ✅ Model structure and parameters
- ✅ Feature coefficients (importance)
- ✅ Confusion matrix plots
- ✅ ROC curve
- ✅ Precision-Recall curve
- ✅ Feature importance bar chart
- ✅ Saves visualization as `model_visualization.png`

---

## Method 2: Run Jupyter Notebook (Best for Interactive Viewing)

### Step 1: Start Jupyter
```powershell
jupyter notebook
```

### Step 2: Open SMOTE Model Notebook
- Navigate to `notebooks/06_smote_model.ipynb`
- Run all cells to see:
  - Interactive visualizations
  - Confusion matrix plots
  - Classification reports
  - Step-by-step model building

**Advantages:**
- See plots inline
- Modify code interactively
- Export visualizations

---

## Method 3: View Model Details (Text Output)

### See Model Structure and Coefficients
```powershell
python -c "import sys; sys.path.append('src'); from sklearn.model_selection import train_test_split; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; from imblearn.pipeline import Pipeline as ImbPipeline; from imblearn.over_sampling import SMOTE; from data_processing.data_loader import load_mammography_data; X, y = load_mammography_data(); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y); pipeline = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)), ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))]); pipeline.fit(X_train, y_train); print('Model Structure:'); [print(f'{i+1}. {name}: {type(step).__name__}') for i, (name, step) in enumerate(pipeline.steps)]; print('\nFeature Coefficients:'); feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness']; [print(f'{name:15s}: {coef:10.4f}') for name, coef in zip(feature_names, pipeline.named_steps['classifier'].coef_[0])]; print(f'\nIntercept: {pipeline.named_steps['classifier'].intercept_[0]:.4f}')"
```

---

## Method 4: Streamlit Dashboard (Interactive Web Interface)

### Start the Dashboard
```powershell
streamlit run streamlit_app/app.py
```

**This will open a web browser where you can:**
- 📊 View data overview
- 🔍 See model performance metrics
- 🎯 Make predictions interactively
- 📈 View visualizations

---

## Method 5: Inspect Saved Model (If Model is Saved)

### Check if Model Exists
```powershell
dir models\
```

### Load and Inspect Model
```powershell
python -c "import joblib; import os; model_path = 'models/model.pkl'; print('Model file exists:', os.path.exists(model_path)); model = joblib.load(model_path) if os.path.exists(model_path) else None; print('Model loaded:', model is not None)"
```

---

## Method 6: View Model Performance Metrics

### Run Model and See All Metrics
```powershell
python run_smote_model.py
```

**Shows:**
- Confusion matrix
- Classification report
- Recall, Precision, F1-Score
- Accuracy

---

## 📊 What You Can See About the Model

### 1. **Model Structure**
- Pipeline steps (StandardScaler → SMOTE → LogisticRegression)
- Hyperparameters (C, max_iter, class_weight, etc.)

### 2. **Feature Importance**
- Logistic Regression coefficients
- Which features are most important
- Positive vs negative impact

### 3. **Performance Visualizations**
- Confusion Matrix (counts and percentages)
- ROC Curve (discrimination ability)
- Precision-Recall Curve (imbalanced performance)
- Feature importance charts

### 4. **Model Metrics**
- Accuracy
- Recall (Sensitivity)
- Precision
- F1-Score
- ROC-AUC
- PR-AUC

---

## 🎨 Quick Visualization Commands

### See Model Coefficients Only
```powershell
python -c "import sys; sys.path.append('src'); from data_processing.data_loader import load_mammography_data; from sklearn.model_selection import train_test_split; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression; from imblearn.pipeline import Pipeline as ImbPipeline; from imblearn.over_sampling import SMOTE; X, y = load_mammography_data(); X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y); pipe = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)), ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))]); pipe.fit(X_train, y_train); print('Feature Coefficients:'); [print(f'{name:15s}: {coef:10.4f}') for name, coef in zip(['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness'], pipe.named_steps['classifier'].coef_[0])]"
```

### See Model Performance Summary
```powershell
python run_smote_model.py
```

---

## 💡 Recommended Approach

**For Quick Viewing:**
```powershell
python visualize_model.py
```

**For Interactive Exploration:**
```powershell
jupyter notebook
# Then open notebooks/06_smote_model.ipynb
```

**For Web Dashboard:**
```powershell
streamlit run streamlit_app/app.py
```

---

## 📁 Files Created

- `visualize_model.py` - Complete visualization script
- `model_visualization.png` - Saved visualization (after running script)

---

## 🔍 Understanding What You See

### Model Coefficients
- **Positive values**: Increase probability of malignant class
- **Negative values**: Decrease probability of malignant class
- **Larger absolute values**: Stronger influence on prediction

### Confusion Matrix
- Shows correct and incorrect predictions
- Helps understand model errors

### ROC Curve
- Shows model's ability to distinguish between classes
- Higher AUC = better discrimination

### Precision-Recall Curve
- Better for imbalanced datasets
- Shows trade-off between precision and recall

