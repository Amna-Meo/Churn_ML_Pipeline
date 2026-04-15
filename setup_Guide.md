# Setup Guide - Customer Churn Prediction Pipeline

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

### 1. Navigate to Project

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

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Install API Dependencies

```bash
pip install -r requirements-api.txt
```

## Project Structure

```
Churn_ML_Pipeline/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── train.py            # Model training & GridSearchCV
│   ├── evaluate.py         # Evaluation metrics
│   └── pipeline.py         # Main pipeline orchestrator
├── api/
│   └── main.py             # FastAPI server
├── models/
│   └── churn_pipeline.pkl  # Trained model
├── main.py                 # CLI entry point
├── requirements.txt
├── requirements-api.txt
├── README.md
└── setup_Guide.md
```

## Usage

### CLI Commands

```bash
# Run full pipeline
python main.py

# Save versioned model (with timestamp)
python main.py --version

# Run inference on CSV
python main.py --input data/sample.csv
```

### Python API

```python
from src.preprocessing import clean_data, engineer_features
from src.pipeline import load_and_predict

# Load and predict
predictions, probabilities = load_and_predict('data/sample.csv')
```

### FastAPI Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer",
    "MonthlyCharges": 45.30,
    "TotalCharges": 1087.20
  }'
```

## Troubleshooting

### Model file not found
Run `python main.py` first to generate the model.

### Import errors
Ensure virtual environment is activated: `source venv/bin/activate`

### Port already in use
Change port: `python -m uvicorn api.main:app --port 8001`
