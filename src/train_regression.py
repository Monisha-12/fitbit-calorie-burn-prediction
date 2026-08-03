"""
src/train_regression.py
=========================
Task 1: Calorie burn prediction (regression). Trains and compares
Linear/Ridge/Lasso, KNN, Decision Tree, Random Forest, XGBoost, and SVR.
Writes results to reports/regression_model_results.csv.

Usage (from repo root):
    python src/train_regression.py --data data/raw/Fitbit_dataset.csv --tune
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — skipping XGBRegressor. "
                   "Install with: pip install xgboost")

from data_preprocessing import prepare, LEAK_OR_ID_COLS

RANDOM_STATE = 42


def build_pipeline(model, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.1, random_state=RANDOM_STATE),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVR": SVR(kernel="rbf", C=100, epsilon=0.5),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    return models


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def tune_random_forest(pipeline, X_train, y_train):
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5],
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best RF params:", search.best_params_)
    return search.best_estimator_


def main(data_path: str, reports_dir: str = "reports", tune: bool = False):
    os.makedirs(reports_dir, exist_ok=True)
    df, target = prepare(data_path)

    numeric_features = [
        "Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage",
        "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
        "Water_Intake (liters)", "Workout_Frequency (days/week)", "Experience_Level",
        "HR_Reserve_Ratio",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns and c not in LEAK_OR_ID_COLS]
    categorical_features = [c for c in ["Gender", "Workout_Type"] if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = get_models()
    results, fitted = [], {}

    for name, model in models.items():
        pipe = build_pipeline(model, numeric_features, categorical_features)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["Model"] = name
        results.append(metrics)
        fitted[name] = pipe
        print(f"{name:20s} MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).set_index("Model").sort_values("R2", ascending=False)
    out_path = os.path.join(reports_dir, "regression_model_results.csv")
    results_df.to_csv(out_path)
    print(f"\nSaved {out_path}")

    best_name = results_df.index[0]
    print(f"Best model: {best_name} (R2={results_df.loc[best_name, 'R2']:.4f})")

    if tune and best_name == "Random Forest":
        print("Tuning Random Forest via GridSearchCV...")
        best_pipe = tune_random_forest(fitted["Random Forest"], X_train, y_train)
        preds = best_pipe.predict(X_test)
        print("Tuned RF metrics:", evaluate(y_test, preds))

    return results_df, fitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Fitbit_dataset.csv")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    main(args.data, reports_dir=args.reports_dir, tune=args.tune)