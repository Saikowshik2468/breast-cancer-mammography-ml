# How to Use the Trained Model - Step-by-Step Guide

## 🎯 Three Ways to Use Your Trained Model

---

## Method 1: Python Script (Quick & Simple) ⚡

### Step-by-Step Instructions:

#### Step 1: Create a prediction script
Create a file called `make_prediction.py` with this code:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from data_processing.data_loader import load_mammography_data

# Load data and train model (or load saved model)
print("Loading data and training model...")
X, y = load_mammography_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
print("Model trained!")

# Example: Make prediction on new data
# Format: [radius, texture, perimeter, area, smoothness, compactness]
new_patient = np.array([[10.0, 15.0, 60.0, 300.0, 0.10, 0.20]])

prediction = pipeline.predict(new_patient)[0]
probability = pipeline.predict_proba(new_patient)[0]

print("\n" + "="*70)
print("PREDICTION RESULT")
print("="*70)
print(f"\nInput Features:")
print(f"  Radius:      {new_patient[0][0]:.2f}")
print(f"  Texture:     {new_patient[0][1]:.2f}")
print(f"  Perimeter:  {new_patient[0][2]:.2f}")
print(f"  Area:        {new_patient[0][3]:.2f}")
print(f"  Smoothness:  {new_patient[0][4]:.2f}")
print(f"  Compactness: {new_patient[0][5]:.2f}")

print(f"\nPrediction: {'MALIGNANT (Cancer Detected)' if prediction == 1 else 'BENIGN (No Cancer)'}")
print(f"\nProbabilities:")
print(f"  Benign:    {probability[0]:.4f} ({probability[0]*100:.2f}%)")
print(f"  Malignant: {probability[1]:.4f} ({probability[1]*100:.2f}%)")
print("="*70)
```

#### Step 2: Run the script
```powershell
cd "F:\ML Project - Breast cancer detect"
python make_prediction.py
```

#### Step 3: Modify the input values
Edit `new_patient` in the script with your actual feature values and run again.

---

## Method 2: Jupyter Notebook (Interactive & Visual) 📓

### Step-by-Step Instructions:

#### Step 1: Start Jupyter Notebook
```powershell
cd "F:\ML Project - Breast cancer detect"
jupyter notebook
```

#### Step 2: Create a new notebook or open existing one
- Click "New" → "Python 3" to create new notebook
- OR open `notebooks/06_smote_model.ipynb` and add a new cell

#### Step 3: Add this code in a new cell:

```python
import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent / 'src'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from data_processing.data_loader import load_mammography_data

# Load and train model
X, y = load_mammography_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
pipeline.fit(X_train, y_train)
```

#### Step 4: Add prediction cell:

```python
# Enter patient features here
radius = 10.0
texture = 15.0
perimeter = 60.0
area = 300.0
smoothness = 0.10
compactness = 0.20

# Make prediction
new_patient = np.array([[radius, texture, perimeter, area, smoothness, compactness]])
prediction = pipeline.predict(new_patient)[0]
probability = pipeline.predict_proba(new_patient)[0]

# Display results
print(f"Prediction: {'MALIGNANT' if prediction == 1 else 'BENIGN'}")
print(f"Benign Probability: {probability[0]:.2%}")
print(f"Malignant Probability: {probability[1]:.2%}")
```

#### Step 5: Run cells
- Press `Shift + Enter` to run each cell
- Modify feature values and re-run to test different patients

---

## Method 3: Streamlit Dashboard (Easiest - No Coding!) 🌐

### Step-by-Step Instructions:

#### Step 1: Navigate to project directory
```powershell
cd "F:\ML Project - Breast cancer detect"
```

#### Step 2: Start Streamlit app
```powershell
streamlit run streamlit_app/app.py
```

#### Step 3: Browser opens automatically
- If not, go to: `http://localhost:8501`

#### Step 4: Use the dashboard
1. **Click on "Model Prediction" tab** (in sidebar)
2. **Enter patient features:**
   - Radius (e.g., 10.0)
   - Texture (e.g., 15.0)
   - Perimeter (e.g., 60.0)
   - Area (e.g., 300.0)
   - Smoothness (e.g., 0.10)
   - Compactness (e.g., 0.20)
3. **Click "Predict" button**
4. **See results:**
   - Prediction (Benign/Malignant)
   - Probability percentages
   - Visual bar chart

#### Step 5: View other tabs
- **Data Overview**: See dataset statistics
- **Model Performance**: View evaluation metrics

#### Step 6: Stop the app
- Press `Ctrl + C` in terminal to stop

---

## 📋 Quick Comparison

| Method | Difficulty | Best For | Visualizations |
|--------|-----------|----------|----------------|
| **Python Script** | Easy | Quick predictions, automation | Text output |
| **Jupyter Notebook** | Medium | Experimentation, analysis | Charts & plots |
| **Streamlit Dashboard** | Easiest | Non-technical users, demos | Interactive UI |

---

## 🎯 Recommended: Start with Streamlit (Easiest)

### Quick Start Commands:
```powershell
cd "F:\ML Project - Breast cancer detect"
streamlit run streamlit_app/app.py
```

Then:
1. Open browser (auto-opens)
2. Click "Model Prediction"
3. Enter values
4. Click "Predict"
5. See results!

---

## 💡 Example: Making Predictions

### Input Format:
```
[radius, texture, perimeter, area, smoothness, compactness]
```

### Example Patient 1 (Benign):
```
[10.0, 15.0, 60.0, 300.0, 0.10, 0.20]
```

### Example Patient 2 (Malignant):
```
[15.0, 20.0, 90.0, 500.0, 0.15, 0.30]
```

---

## 🔧 Troubleshooting

### Issue: Streamlit not found
**Solution:**
```powershell
pip install streamlit
```

### Issue: Model not found
**Solution:** The model trains automatically when you run the script/notebook/dashboard

### Issue: Port already in use
**Solution:**
```powershell
streamlit run streamlit_app/app.py --server.port 8502
```

---

## 📝 Next Steps

1. **Try Streamlit** (easiest way to use the model)
2. **Experiment** with different feature values
3. **Compare** predictions with actual results
4. **Save** predictions to a file if needed

---

## ✅ You're Ready!

Choose any method above and start making predictions with your trained model!

