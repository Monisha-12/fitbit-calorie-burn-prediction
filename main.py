import sys
import os

sys.path.append(os.path.abspath("."))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    handle_missing_values,
    encode_features,
    cap_outliers,
    scale_features
)

from src.train_regression import train_and_compare_models

from src.train_clustering import train_clustering_pipeline, save_cluster_plot

def main():
    print("🚀 Starting Fitbit ML Pipeline...")

    # 1. Load data
    df = pd.read_csv("data/raw/Fitbit_dataset.csv")

    # 2. Clean column names
    df.columns = df.columns.str.strip()

    # 3. Drop unwanted index column
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # 4. 🔥 DROP LEAKAGE FEATURES
    leakage_cols = ["Base_MET", "Effective_MET", "HR_Intensity"]

    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    print("Dropped leakage features:", leakage_cols)

    # 5. Preprocessing
    df = handle_missing_values(df)
    df = cap_outliers(df)
    df = encode_features(df)

    print("Columns after encoding:")
    print(df.columns.tolist())

    # 6. Target
    target_col = "Calories_Burned (kcal)"

    X = df.drop(target_col, axis=1).astype(float)
    y = df[target_col]

    # 7. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 9. Train + Compare models
    results_df = train_and_compare_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n📊 Model Comparison Results:")
    print(results_df)

    # 10. Save results
    results_df.to_csv("data/reports/regression_model_results.csv", index=False)

    print("\n📁 Results saved to data/reports/regression_model_results.csv")
    print("✅ Pipeline completed!")

    print("Final columns used for training:")
    print(X.columns.tolist())

    # =========================
    # CLUSTERING TASK
    # =========================
    print("\n🔍 Starting Clustering Pipeline...")

    clustering_results = train_clustering_pipeline(df, n_clusters=3)

    print(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}")
    print("\nCluster Size Distribution:")
    print(clustering_results["cluster_sizes"])

    print("\nCluster Feature Means:")
    print(clustering_results["cluster_feature_means"])

    # Save clustering outputs
    os.makedirs("data/reports", exist_ok=True)
    os.makedirs("data/visuals", exist_ok=True)

    clustering_results["cluster_feature_means"].to_csv(
        "data/reports/cluster_feature_means.csv"
    )

    clustering_results["df_with_clusters"].to_csv(
        "data/reports/clustered_fitbit_data.csv",
        index=False
    )

    save_cluster_plot(
        clustering_results["pca_data"],
        clustering_results["cluster_labels"],
        output_path="data/visuals/pca_clusters.png"
    )

    print("\n📁 Clustering results saved:")
    print("- data/reports/cluster_feature_means.csv")
    print("- data/reports/clustered_fitbit_data.csv")
    print("- data/visuals/pca_clusters.png")

if __name__ == "__main__":
    main()