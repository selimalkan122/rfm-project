from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def prepare_rfm_for_clustering(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform skewed RFM features and standardise them for K-Means."""
    X = rfm[["recency", "frequency", "monetary"]].copy()
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"] = np.log1p(X["monetary"])
    X["recency"] = np.log1p(X["recency"])

    scaler = StandardScaler()
    return scaler.fit_transform(X)


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """Run K-Means across k values and return inertia + silhouette curves."""
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))

    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def label_kmeans_clusters(cluster_profile: pd.DataFrame) -> dict:
    """Auto-label clusters using monetary (desc) and recency (asc) ranking."""
    labels_pool = [
        "High-Value Active",
        "Mid-Value Regulars",
        "Low-Value At-Risk",
        "Lapsed Customers",
        "Occasional Shoppers",
    ]

    sorted_idx = (
        cluster_profile.sort_values(["avg_monetary", "avg_recency"], ascending=[False, True])
        .index.tolist()
    )

    return {
        cluster: labels_pool[i % len(labels_pool)]
        for i, cluster in enumerate(sorted_idx)
    }

