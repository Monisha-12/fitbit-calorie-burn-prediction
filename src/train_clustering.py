import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [
        "Calories_Burned (kcal)",
        "Workout_Type",
        "Base_MET",
        "Effective_MET",
        "HR_Intensity",
    ]

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Encode remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df


def run_kmeans_clustering(X_scaled, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)

    return model, cluster_labels, score


def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    return pca, X_pca, explained_variance


def get_cluster_summary(df_original, cluster_labels):
    df_summary = df_original.copy()
    df_summary["Cluster"] = cluster_labels

    cluster_sizes = df_summary["Cluster"].value_counts().sort_index()

    numeric_summary = (
        df_summary.groupby("Cluster")
        .mean(numeric_only=True)
        .round(2)
    )

    return cluster_sizes, numeric_summary, df_summary


def save_cluster_plot(X_pca, cluster_labels, output_path="visuals/pca_clusters.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
    plt.title("PCA-Based KMeans Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_clustering_pipeline(df: pd.DataFrame, n_clusters=3):
    # Keep original for interpretation
    df_original = df.copy()

    # Prepare data
    X = prepare_clustering_data(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca, X_pca, explained_variance = apply_pca(X_scaled, n_components=2)

    # KMeans
    kmeans_model, cluster_labels, sil_score = run_kmeans_clustering(
        X_scaled, n_clusters=n_clusters
    )

    # Summaries
    cluster_sizes, cluster_feature_means, df_with_clusters = get_cluster_summary(
        df_original, cluster_labels
    )

    return {
        "prepared_data": X,
        "scaled_data": X_scaled,
        "pca_model": pca,
        "pca_data": X_pca,
        "explained_variance": explained_variance,
        "kmeans_model": kmeans_model,
        "cluster_labels": cluster_labels,
        "silhouette_score": sil_score,
        "cluster_sizes": cluster_sizes,
        "cluster_feature_means": cluster_feature_means,
        "df_with_clusters": df_with_clusters,
    }