"""
main.py
========
Single entry point for the Fitbit calorie burn prediction + workout
clustering project. Runs from the repo root.

Usage:
    python main.py --task regression --data data/raw/Fitbit_dataset.csv
    python main.py --task clustering --data data/raw/Fitbit_dataset.csv
    python main.py --task all         --data data/raw/Fitbit_dataset.csv --tune
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import prepare  # noqa: E402


def run_regression(data_path, reports_dir, tune):
    import train_regression
    train_regression.main(data_path, reports_dir=reports_dir, tune=tune)


def run_clustering(data_path, reports_dir, visuals_dir, include_calories):
    import train_clustering
    train_clustering.main(data_path, reports_dir=reports_dir,
                           visuals_dir=visuals_dir, include_calories=include_calories)


def save_processed_snapshot(data_path, processed_path):
    df, _ = prepare(data_path)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned snapshot to {processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fitbit calorie prediction + workout clustering")
    parser.add_argument("--task", choices=["regression", "clustering", "all"], default="all")
    parser.add_argument("--data", default="data/raw/Fitbit_dataset.csv")
    parser.add_argument("--processed", default="data/processed/fitbit_clean.csv")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--visuals_dir", default="visuals")
    parser.add_argument("--tune", action="store_true", help="Tune Random Forest if it wins (regression)")
    parser.add_argument("--include_calories", action="store_true",
                         help="Include Calories_Burned as a clustering feature")
    args = parser.parse_args()

    save_processed_snapshot(args.data, args.processed)

    if args.task in ("regression", "all"):
        print("\n" + "=" * 60)
        print("TASK 1: REGRESSION")
        print("=" * 60)
        run_regression(args.data, args.reports_dir, args.tune)

    if args.task in ("clustering", "all"):
        print("\n" + "=" * 60)
        print("TASK 2: CLUSTERING")
        print("=" * 60)
        run_clustering(args.data, args.reports_dir, args.visuals_dir, args.include_calories)