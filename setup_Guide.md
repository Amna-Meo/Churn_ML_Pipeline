# Setup Guide - Customer Churn Prediction Pipeline

## Prerequisites

- Python 3.8+
- pip package manager
- Git (optional)

## Installation

### 1. Clone or Navigate to Project

```bash
cd churn-pipeline-ML
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
churn-pipeline-ML/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── src/
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── train.py                 # Model training & GridSearchCV
│   ├── evaluate.py              # Evaluation metrics
│   ├── pipeline.py              # Main pipeline orchestrator
│   └── inference.py             # Inference script
├── models/
│   └── churn_pipeline.pkl       # Trained model (after running pipeline)
├── requirements.txt
├── README.md
└── setup_Guide.md
```

## Usage

### Run the Complete Pipeline

```bash
source venv/bin/activate
python -m src.pipeline
```

This will:
1. Load and clean the dataset
2. Engineer new features
3. Identify feature types (numerical/categorical)
4. Build preprocessing pipeline
5. Split data into train/test sets
6. Train Logistic Regression and Random Forest models
7. Tune hyperparameters using GridSearchCV
8. Evaluate the best model
9. Export the pipeline to `models/churn_pipeline.pkl`

### Run Inference

```bash
python -m src.inference
```

### Load Model in Python

```python
import joblib
import pandas as pd
from src.data_preprocessing import clean_data, engineer_features

# Load trained model
model = joblib.load('models/churn_pipeline.pkl')

# Load and prepare data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = clean_data(df)
df = engineer_features(df)
X = df.drop('Churn', axis=1)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

## Dataset

The pipeline uses the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Features:**
- Customer demographics (gender, SeniorCitizen, Partner, Dependents)
- Services (PhoneService, InternetService, etc.)
- Account info (tenure, Contract, PaymentMethod)
- Billing (MonthlyCharges, TotalCharges)

**Target:** Churn (Yes/No)

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical operations |
| scikit-learn | >=1.3.0 | ML pipeline & models |
| joblib | >=1.3.0 | Model serialization |

## Troubleshooting

### Permission Denied (venv activation)
```bash
chmod +x venv/bin/activate
```

### Package Installation Failed
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Model File Not Found
Run the pipeline first to generate `models/churn_pipeline.pkl`.

## Development

### Run Tests
```bash
source venv/bin/activate
pytest
```

### Deactivate Environment
```bash
deactivate
```

## Notes

- The pipeline uses balanced class weights to handle churn imbalance
- Training uses parallel processing (`n_jobs=-1`) for faster execution
- The exported `.pkl` file includes both preprocessing and model
