"""
src/train_clustering.py
=========================
Task 2: Workout pattern clustering (unsupervised). Encodes -> scales -> PCA
-> KMeans, evaluated with Silhouette Score, plus Hierarchical/DBSCAN for
comparison. Writes:
    reports/clustered_fitbit_data.csv   (original rows + Cluster column)
    reports/cluster_feature_means.csv   (centroid means per cluster)
    visuals/pca_clusters.png
    visuals/k_selection.png
    visuals/silhouette_plot.png
    visuals/cluster_profiles_heatmap.png

Usage (from repo root):
    python src/train_clustering.py --data data/raw/Fitbit_dataset.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples

from data_preprocessing import prepare, LEAK_OR_ID_COLS

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def build_feature_set(df: pd.DataFrame, target: str, include_calories: bool = False):
    exclude = set(LEAK_OR_ID_COLS) | {"Workout_Type"}
    if not include_calories:
        exclude.add(target)

    numeric_features = [
        "Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage",
        "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
        "Water_Intake (liters)", "Workout_Frequency (days/week)", "Experience_Level",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns and c not in exclude]
    categorical_features = [c for c in ["Gender"] if c in df.columns and c not in exclude]
    return numeric_features, categorical_features


def preprocess_for_clustering(df, numeric_features, categorical_features, n_components=0.90):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    X_scaled = preprocessor.fit_transform(df[numeric_features + categorical_features])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} components explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of variance")
    return X_scaled, X_pca, pca


def sweep_k(X_pca, k_range=range(2, 9)):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))
        print(f"k={k}: inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.4f}")
    return list(k_range), inertias, sil_scores


def plot_k_selection(k_range, inertias, sil_scores, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_range, inertias, marker="o")
    axes[0].set_title("Elbow Method (Inertia vs k)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, sil_scores, marker="o", color="darkorange")
    axes[1].axhline(0.15, color="red", linestyle="--", label="Acceptance threshold (0.15)")
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "k_selection.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fit_alternative_methods(X_pca, k):
    results = {}
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_pca)
    results["Hierarchical"] = (agg_labels, silhouette_score(X_pca, agg_labels))

    best_db, best_score, best_eps = None, -1, None
    for eps in np.arange(0.3, 2.0, 0.1):
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        try:
            score = silhouette_score(X_pca, labels)
        except ValueError:
            continue
        if score > best_score:
            best_db, best_score, best_eps = labels, score, eps
    if best_db is not None:
        results["DBSCAN"] = (best_db, best_score)
        print(f"DBSCAN best eps={best_eps:.1f}, silhouette={best_score:.4f}")
    else:
        warnings.warn("DBSCAN did not find >=2 clusters in the eps range tried")
    return results


def plot_pca_clusters(X_pca, labels, out_dir):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans Clusters (PCA space)")
    plt.colorbar(scatter, label="Cluster")
    path = os.path.join(out_dir, "pca_clusters.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_silhouette_diagram(X_pca, labels, out_dir):
    sil_vals = silhouette_samples(X_pca, labels)
    k = len(set(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10
    for i in range(k):
        cluster_vals = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(cluster_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i))
        y_lower = y_upper + 10
    ax.axvline(silhouette_score(X_pca, labels), color="red", linestyle="--", label="Mean score")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot per Cluster")
    ax.legend()
    path = os.path.join(out_dir, "silhouette_plot.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def interpret_clusters(df, numeric_features, labels, workout_col, exp_col, visuals_dir, reports_dir):
    df = df.copy()
    df["Cluster"] = labels

    print("\n=== Cluster sizes ===")
    print(df["Cluster"].value_counts().sort_index())

    centroid_means = df.groupby("Cluster")[numeric_features].mean().round(2)
    print("\n=== Cluster feature means ===")
    print(centroid_means)
    means_path = os.path.join(reports_dir, "cluster_feature_means.csv")
    centroid_means.to_csv(means_path)
    print(f"Saved {means_path}")

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(numeric_features))))
    sns.heatmap(
        (centroid_means - centroid_means.mean()) / centroid_means.std(),
        cmap="coolwarm", center=0, annot=centroid_means.values, fmt=".1f", ax=ax,
    )
    ax.set_title("Cluster Profiles (z-scored coloring, raw values annotated)")
    heatmap_path = os.path.join(visuals_dir, "cluster_profiles_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved {heatmap_path}")

    if workout_col and workout_col in df.columns:
        print(f"\n=== Cluster vs {workout_col} (row %) ===")
        print((pd.crosstab(df["Cluster"], df[workout_col], normalize="index") * 100).round(1))

    if exp_col and exp_col in df.columns:
        print(f"\n=== Cluster vs {exp_col} (mean) ===")
        print(df.groupby("Cluster")[exp_col].mean().round(2))

    clustered_path = os.path.join(reports_dir, "clustered_fitbit_data.csv")
    df.to_csv(clustered_path, index=False)
    print(f"Saved {clustered_path}")

    return centroid_means


def main(data_path: str, reports_dir="reports", visuals_dir="visuals", include_calories=False):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    df, target = prepare(data_path)
    numeric_features, categorical_features = build_feature_set(df, target, include_calories)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    X_scaled, X_pca, pca = preprocess_for_clustering(df, numeric_features, categorical_features)

    k_range, inertias, sil_scores = sweep_k(X_pca, range(2, 9))
    plot_k_selection(k_range, inertias, sil_scores, visuals_dir)

    best_k = k_range[int(np.argmax(sil_scores))]
    print(f"\nBest k by silhouette score: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"Final KMeans (k={best_k}) silhouette score: {score:.4f} "
          f"({'PASS' if score >= 0.15 else 'BELOW'} 0.15 threshold)")

    plot_pca_clusters(X_pca, labels, visuals_dir)
    plot_silhouette_diagram(X_pca, labels, visuals_dir)

    workout_col = "Workout_Type" if "Workout_Type" in df.columns else None
    exp_col = "Experience_Level" if "Experience_Level" in df.columns else None
    interpret_clusters(df, numeric_features, labels, workout_col, exp_col, visuals_dir, reports_dir)

    print("\n=== Alternative clustering methods (for comparison) ===")
    alt_results = fit_alternative_methods(X_pca, best_k)
    for name, (_, alt_score) in alt_results.items():
        print(f"{name}: silhouette={alt_score:.4f}")

    return {"labels": labels, "silhouette": score, "pca": pca, "alt_results": alt_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Fitbit_dataset.csv")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--visuals_dir", default="visuals")
    parser.add_argument("--include_calories", action="store_true")
    args = parser.parse_args()
    main(args.data, reports_dir=args.reports_dir, visuals_dir=args.visuals_dir,
         include_calories=args.include_calories)