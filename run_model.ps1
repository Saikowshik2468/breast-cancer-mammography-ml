# Breast Cancer Detection - Model Execution Script
# This script helps you run the ML models easily

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Breast Cancer Detection - Model Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$packages = @("sklearn", "imblearn", "pandas", "numpy", "jupyter", "streamlit")
$missing = @()

foreach ($pkg in $packages) {
    $result = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
} else {
    Write-Host "All dependencies are installed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Select an option:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Run Jupyter Notebook (Interactive)" -ForegroundColor White
Write-Host "2. Run Baseline Model Training" -ForegroundColor White
Write-Host "3. Run SMOTE Model Training" -ForegroundColor White
Write-Host "4. Run Streamlit Dashboard" -ForegroundColor White
Write-Host "5. Run Class Imbalance Analysis" -ForegroundColor White
Write-Host "6. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "Starting Jupyter Notebook..." -ForegroundColor Green
        jupyter notebook
    }
    "2" {
        Write-Host "Running Baseline Model..." -ForegroundColor Green
        python -c "import sys; sys.path.append('src'); from models.train import main; main()"
    }
    "3" {
        Write-Host "Opening SMOTE Model Notebook..." -ForegroundColor Green
        jupyter notebook notebooks/06_smote_model.ipynb
    }
    "4" {
        Write-Host "Starting Streamlit Dashboard..." -ForegroundColor Green
        Write-Host "The dashboard will open in your browser..." -ForegroundColor Yellow
        streamlit run streamlit_app/app.py
    }
    "5" {
        Write-Host "Running Class Imbalance Analysis..." -ForegroundColor Green
        python analyze_class_imbalance.py
    }
    "6" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit
    }
    default {
        Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
    }
}

