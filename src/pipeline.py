import os
import logging
import joblib
import pandas as pd
from datetime import datetime

from src.preprocessing import (
    get_feature_types,
    build_preprocessing_pipeline,
    clean_data,
    engineer_features,
)
from src.train import train_models
from src.evaluate import evaluate_model
from src.train import train_models
from src.evaluate import evaluate_model

logger = logging.getLogger(__name__)


def run_pipeline(versioned=False):
    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)

    logger.info("[1/7] Loading dataset...")
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    logger.info(f"Dataset shape: {df.shape}")

    logger.info("[2/7] Cleaning data...")
    df = clean_data(df)
    logger.info(f"Cleaned dataset shape: {df.shape}")

    logger.info("[2.5/7] Engineering features...")
    df = engineer_features(df)
    logger.info(f"Feature engineered dataset shape: {df.shape}")

    logger.info("[3/7] Identifying feature types...")
    numerical_features, categorical_features = get_feature_types(df)
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(
        f"Categorical features ({len(categorical_features)}): {categorical_features}"
    )

    logger.info("[4/7] Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features
    )
    logger.info("Preprocessing pipeline created successfully!")

    logger.info("[5/7] Splitting data...")
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    logger.info("[6/7] Training and tuning models...")
    best_model, best_model_name, results = train_models(X_train, y_train, preprocessor)

    logger.info("[7/7] Evaluating best model...")
    metrics = evaluate_model(best_model, X_test, y_test)

    logger.info("[8/8] Exporting pipeline...")
    os.makedirs("models", exist_ok=True)

    if versioned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/churn_pipeline_{timestamp}.pkl"
    else:
        model_path = "models/churn_pipeline.pkl"

    joblib.dump(best_model, model_path)
    logger.info(f"Pipeline exported to: {model_path}")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)

    return model_path, metrics


def load_and_predict(input_path, model_path="models/churn_pipeline.pkl"):
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    df = clean_data(df)
    df = engineer_features(df)
    X = df.drop("Churn", axis=1)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    logger.info(f"Predictions complete. Total: {len(predictions)}")
    logger.info(f"Churn predicted: {(predictions == 1).sum()}")
    logger.info(f"No churn predicted: {(predictions == 0).sum()}")

    return predictions, probabilities


if __name__ == "__main__":
    run_pipeline()
