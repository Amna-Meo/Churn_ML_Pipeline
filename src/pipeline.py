import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_preprocessing import (
    get_feature_types,
    build_preprocessing_pipeline,
    clean_data,
    engineer_features,
)
from src.train import train_models, export_pipeline
from src.evaluate import evaluate_model


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 60)

    print("\n[1/7] Loading dataset...")
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Dataset shape: {df.shape}")

    print("\n[2/7] Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned dataset shape: {df.shape}")

    print("\n[2.5/7] Engineering features...")
    df = engineer_features(df)
    print(f"Feature engineered dataset shape: {df.shape}")

    print("\n[3/7] Identifying feature types...")
    numerical_features, categorical_features = get_feature_types(df)
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    print("\n[4/7] Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features
    )
    print("Preprocessing pipeline created successfully!")

    print("\n[5/7] Splitting data...")
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\n[6/7] Training and tuning models...")
    best_model, best_model_name, results = train_models(X_train, y_train, preprocessor)

    print("\n[7/7] Evaluating best model...")
    metrics = evaluate_model(best_model, X_test, y_test)

    print("\n[8/7] Exporting pipeline...")
    os.makedirs("models", exist_ok=True)
    export_pipeline(best_model, "models/churn_pipeline.pkl")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

    return best_model, metrics


if __name__ == "__main__":
    main()
