"""
src/data_preprocessing.py
==========================
Shared cleaning, outlier-capping, and feature-set logic used by both
train_regression.py and train_clustering.py, so the two training scripts
stay in sync on how the raw Fitbit data is prepared.
"""

import numpy as np
import pandas as pd

# Columns that are either row-index artifacts or deterministically build
# the target (Calories_Burned = Effective_MET * Weight * Session_Duration,
# verified during EDA — corr ~0.99999997), so they must never be used as
# model features.
LEAK_OR_ID_COLS = {"Unnamed: 0", "Base_MET", "HR_Intensity", "Effective_MET"}


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Fitbit CSV and strip whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (median/mode) and drop exact duplicate rows."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} duplicate rows")

    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 1.5) -> pd.DataFrame:
    """Clip values outside [Q1 - factor*IQR, Q3 + factor*IQR] rather than dropping rows."""
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[c] = df[c].clip(lower, upper)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add HR_Reserve_Ratio (exercise-intensity proxy) and repair invalid BMI."""
    df = df.copy()

    if {"Max_BPM", "Avg_BPM", "Resting_BPM"}.issubset(df.columns):
        denom = (df["Max_BPM"] - df["Resting_BPM"]).replace(0, np.nan)
        df["HR_Reserve_Ratio"] = (df["Avg_BPM"] - df["Resting_BPM"]) / denom
        df["HR_Reserve_Ratio"] = df["HR_Reserve_Ratio"].fillna(df["HR_Reserve_Ratio"].median())

    if {"Weight (kg)", "Height (m)", "BMI"}.issubset(df.columns):
        mask = (df["BMI"].isnull()) | (df["BMI"] <= 0)
        if mask.any():
            df.loc[mask, "BMI"] = df.loc[mask, "Weight (kg)"] / (df.loc[mask, "Height (m)"] ** 2)

    return df


def get_target_column(df: pd.DataFrame) -> str:
    """Detect the calories target column, whichever export naming was used."""
    candidates = [c for c in df.columns if c.startswith("Calories_Burned")]
    if not candidates:
        raise ValueError("No Calories_Burned column found in dataset")
    return candidates[0]


def base_numeric_features(df: pd.DataFrame) -> list:
    cols = [
        "Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage",
        "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
        "Water_Intake (liters)", "Workout_Frequency (days/week)", "Experience_Level",
    ]
    return [c for c in cols if c in df.columns and c not in LEAK_OR_ID_COLS]


def prepare(path: str):
    """
    Full shared pipeline: load -> clean -> cap outliers -> engineer features.
    Returns the prepared DataFrame plus the detected target column name.
    """
    df = load_raw(path)
    df = clean(df)
    target = get_target_column(df)
    numeric_cols = base_numeric_features(df)
    df = cap_outliers_iqr(df, numeric_cols)
    df = engineer_features(df)
    return df, target