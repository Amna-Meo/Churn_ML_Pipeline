import joblib
import pandas as pd
from src.data_preprocessing import clean_data, engineer_features


def load_model(filepath="models/churn_pipeline.pkl"):
    return joblib.load(filepath)


def predict(model, df):
    df = clean_data(df)
    df = engineer_features(df)
    X = df.drop("Churn", axis=1)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


def main():
    model = load_model()
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    predictions, probabilities = predict(model, df)

    print("Predictions for dataset:")
    print(f"Total samples: {len(predictions)}")
    print(f"Churn predicted: {(predictions == 1).sum()}")
    print(f"No churn predicted: {(predictions == 0).sum()}")
    print(f"\nAverage churn probability: {probabilities.mean():.2%}")

    return predictions, probabilities


if __name__ == "__main__":
    main()
