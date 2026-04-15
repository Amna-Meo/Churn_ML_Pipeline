import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def get_feature_types(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    if "customerID" in categorical_features:
        categorical_features.remove("customerID")
    if "Churn" in categorical_features:
        categorical_features.remove("Churn")

    return numerical_features, categorical_features


def build_preprocessing_pipeline(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def clean_data(df):
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    return df


def engineer_features(df):
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "mid", "established", "loyal"],
    ).astype(str)

    df["AvgMonthlyCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargesRatio"] = df["MonthlyCharges"] / (df["AvgMonthlyCharges"] + 1)

    internet_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]
    df["SupportServicesCount"] = sum(
        (df[col] == "Yes").astype(int) for col in internet_services
    )

    streaming_services = ["StreamingTV", "StreamingMovies"]
    df["StreamingCount"] = sum(
        (df[col] == "Yes").astype(int) for col in streaming_services
    )

    df["HasPhoneAndInternet"] = (
        (df["PhoneService"] == "Yes") & (df["InternetService"] != "No")
    ).astype(int)

    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    return df
