import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.clip(df[col], lower, upper)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler