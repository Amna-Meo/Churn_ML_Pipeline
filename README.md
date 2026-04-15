# Customer Churn Prediction - End-to-End ML Pipeline

## Problem Statement

Telecommunication companies face significant revenue loss due to customer churn. This system predicts whether a customer will churn based on historical data using a fully automated ML pipeline.

## System Architecture

```
Raw Dataset (CSV)
     ↓
Data Cleaning & Validation
     ↓
Feature Engineering
     ↓
ColumnTransformer (Scaling + Encoding)
     ↓
ML Pipeline (Model + Preprocessing)
     ↓
GridSearchCV (Optimization)
     ↓
Evaluation Metrics
     ↓
Export (.pkl using joblib)
     ↓
FastAPI / CLI / Direct Python
```

## Project Structure

```
Churn_ML_Pipeline/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── preprocessing.py   # Data cleaning & feature engineering
│   ├── train.py           # Model training & GridSearchCV
│   ├── evaluate.py        # Evaluation metrics
│   └── pipeline.py        # Main pipeline orchestrator
├── api/
│   └── main.py           # FastAPI server
├── models/
│   └── churn_pipeline.pkl
├── requirements.txt
├── requirements-api.txt
├── main.py               # CLI entry point
├── README.md
└── setup_Guide.md
```

## Models & Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.73 | 0.50 | 0.81 | 0.61 |
| Random Forest | 0.73 | 0.49 | 0.80 | 0.61 |

**Best Model:** Logistic Regression (F1: 0.61)

## Quick Start

### 1. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
# Standard run
python main.py

# Save versioned model
python main.py --version
```

### 3. Run API Server
```bash
pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload
```

### 4. CLI Predictions
```bash
python main.py --input data/sample.csv
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict_batch` | Batch predictions |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":24,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"No","DeviceProtection":"Yes","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"One year","PaperlessBilling":"No","PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":45.30,"TotalCharges":1087.20}'
```

## Core Components

### Data Preprocessing
- Missing value handling (median/mode)
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Custom feature engineering (tenure groups, service counts)

### Model Training
- **Logistic Regression**: Baseline, interpretable
- **Random Forest**: Non-linear ensemble

### Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- F1 score optimization
- Parallel processing (n_jobs=-1)

### Model Serialization
- Single `.pkl` file includes preprocessing + model
- No need to redo preprocessing
- Direct inference on raw data

## Non-Functional Requirements

| Requirement | Implementation |
|-------------|----------------|
| Performance | n_jobs=-1 for parallel training |
| Reusability | Single pipeline handles everything |
| Scalability | FastAPI-ready for deployment |
| Maintainability | Modular codebase |

## Risks & Limitations

- Class imbalance (~26% churn rate) - mitigated with balanced class weights
- Data leakage prevented by keeping preprocessing inside pipeline
- Random Forest overfitting - controlled via max_depth tuning

## Skills Demonstrated

- ML Pipeline Engineering
- Feature Engineering
- Model Optimization
- Production ML (FastAPI, CLI)
- Code Structuring
- Model Serialization
