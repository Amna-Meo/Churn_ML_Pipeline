# Customer Churn Prediction Pipeline

End-to-end ML pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Project Structure

```
churn-ml-pipeline/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── pipeline.py
│   └── inference.py
├── models/
│   └── churn_pipeline.pkl
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline:

```bash
source venv/bin/activate
python -m src.pipeline
```

### Use the trained model for predictions:

```bash
python -m src.inference
```

### Load model in Python:

```python
import joblib
import pandas as pd
from src.data_preprocessing import clean_data, engineer_features

model = joblib.load('models/churn_pipeline.pkl')
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = clean_data(df)
df = engineer_features(df)
X = df.drop('Churn', axis=1)
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Pipeline Features

- **Data Cleaning**: Handles missing values, type conversions
- **Feature Engineering**: Tenure groups, charges ratio, service counts
- **Preprocessing**: 
  - Numerical: Median imputation + StandardScaler
  - Categorical: Most frequent imputation + OneHotEncoder
- **Models**: Logistic Regression, Random Forest (with balanced class weights)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Export**: Single `.pkl` file including preprocessing + model
- **Parallel Processing**: n_jobs=-1 for faster training

## Requirements

- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- joblib>=1.3.0

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.73   |
| Precision | 0.50   |
| Recall    | 0.81   |
| F1 Score  | 0.61   |
