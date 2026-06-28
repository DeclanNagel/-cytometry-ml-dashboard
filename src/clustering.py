"""
src/clustering.py
------------------
ML-based clustering ("automated gating") and cluster interpretation
utilities for preprocessed cytometry data.

Supports three commonly used unsupervised clustering algorithms:
    - KMeans
    - DBSCAN
    - Gaussian Mixture Models (GMM)

Also provides a lightweight rule-based "auto-gating interpretation"
step that inspects per-cluster median marker expression (relative to
the overall population) and proposes a likely cell-population label,
mimicking the kind of reasoning an analyst applies when manually
gating flow cytometry data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


class ClusteringError(ValueError):
    """Raised when clustering inputs or parameters are invalid."""


# Markers that, when relatively high in a cluster, are characteristic
# of a known immune cell type. Used only for the human-readable
# "auto-gating" interpretation -- not for the clustering itself.
_LINEAGE_MARKER_HINTS: dict[str, str] = {
    "CD3": "T cell",
    "CD4": "CD4+ (helper T cell / monocyte co-marker)",
    "CD8": "CD8+ (cytotoxic T cell)",
    "CD19": "B cell",
    "CD14": "Monocyte",
    "CD56": "NK cell",
    "CD45": "Leukocyte (pan-immune)",
}


def run_clustering(
    df: pd.DataFrame,
    marker_columns: list[str],
    algorithm: str,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Run a clustering algorithm on the selected (preprocessed) marker columns.

    Args:
        df: Preprocessed DataFrame (after scaling/transformation).
        marker_columns: Columns to use as clustering features.
        algorithm: One of "KMeans", "DBSCAN", "Gaussian Mixture".
        **kwargs: Algorithm-specific hyperparameters:
            - KMeans: n_clusters (int), random_state (int)
            - DBSCAN: eps (float), min_samples (int)
            - Gaussian Mixture: n_components (int), random_state (int)

    Returns:
        Tuple of (cluster_labels array, dict of parameters actually used).
        For DBSCAN, noise points are labeled -1 (standard scikit-learn
        convention), and are reported separately in the cluster summary.

    Raises:
        ClusteringError: If algorithm is unrecognized, marker_columns is
            empty, or any required column is missing/non-numeric.
    """
    if not marker_columns:
        raise ClusteringError("At least one marker column must be selected for clustering.")
    missing = [c for c in marker_columns if c not in df.columns]
    if missing:
        raise ClusteringError(f"Columns not found in data: {missing}")

    X = df[marker_columns].to_numpy()
    if not np.isfinite(X).all():
        raise ClusteringError(
            "Input data contains NaN or infinite values. "
            "Please complete preprocessing (missing-value handling) before clustering."
        )

    if algorithm == "KMeans":
        n_clusters = int(kwargs.get("n_clusters", 5))
        random_state = int(kwargs.get("random_state", 42))
        if n_clusters < 1:
            raise ClusteringError("n_clusters must be >= 1.")
        if n_clusters > X.shape[0]:
            raise ClusteringError("n_clusters cannot exceed the number of cells.")
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        params_used = {"algorithm": "KMeans", "n_clusters": n_clusters, "random_state": random_state}

    elif algorithm == "DBSCAN":
        eps = float(kwargs.get("eps", 0.5))
        min_samples = int(kwargs.get("min_samples", 10))
        if eps <= 0:
            raise ClusteringError("eps must be a positive number.")
        if min_samples < 1:
            raise ClusteringError("min_samples must be >= 1.")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        params_used = {"algorithm": "DBSCAN", "eps": eps, "min_samples": min_samples}

    elif algorithm == "Gaussian Mixture":
        n_components = int(kwargs.get("n_components", 5))
        random_state = int(kwargs.get("random_state", 42))
        if n_components < 1:
            raise ClusteringError("n_components must be >= 1.")
        if n_components > X.shape[0]:
            raise ClusteringError("n_components cannot exceed the number of cells.")
        model = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = model.fit_predict(X)
        params_used = {
            "algorithm": "Gaussian Mixture",
            "n_components": n_components,
            "random_state": random_state,
        }

    else:
        raise ClusteringError(
            f"Unknown algorithm '{algorithm}'. Expected one of "
            "['KMeans', 'DBSCAN', 'Gaussian Mixture']."
        )

    return labels, params_used


def build_cluster_summary(
    df: pd.DataFrame,
    marker_columns: list[str],
    cluster_labels: np.ndarray,
    raw_marker_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-cluster summary table: size, proportion, and median marker expression.

    Args:
        df: Preprocessed (e.g. scaled) DataFrame the clustering was run on.
            Used only to align rows with cluster_labels.
        marker_columns: Marker columns to summarize.
        cluster_labels: Cluster assignment for each row in df (same order/length).
        raw_marker_df: Optional DataFrame (same row order as df) containing
            the *original, untransformed* marker values. When provided,
            median expression in the summary is reported on the original
            scale, which is far more interpretable than z-scored values.
            If None, medians are reported on whatever scale `df` is in.

    Returns:
        DataFrame indexed by cluster ID with columns:
            n_cells, percent_of_total, and median_<marker> for each marker.

    Raises:
        ClusteringError: If cluster_labels length does not match df length.
    """
    if len(cluster_labels) != df.shape[0]:
        raise ClusteringError("cluster_labels length must match the number of rows in df.")

    source = raw_marker_df if raw_marker_df is not None else df
    working = source[marker_columns].copy()
    working["cluster"] = cluster_labels

    total_cells = len(cluster_labels)
    grouped = working.groupby("cluster")

    summary = grouped.size().rename("n_cells").to_frame()
    summary["percent_of_total"] = (summary["n_cells"] / total_cells * 100).round(2)

    medians = grouped[marker_columns].median()
    medians.columns = [f"median_{c}" for c in medians.columns]

    summary = summary.join(medians)
    summary = summary.sort_values("n_cells", ascending=False)
    summary.index.name = "cluster"
    return summary.reset_index()


def interpret_clusters(
    cluster_summary: pd.DataFrame,
    marker_columns: list[str],
) -> pd.DataFrame:
    """Generate a simple rule-based "likely population" note for each cluster.

    For each cluster, this compares its median expression of each known
    lineage marker (CD3, CD4, CD8, CD19, CD14, CD56, CD45) against the
    population-wide median for that marker. Markers more than a small
    margin above the overall median are treated as "high" / positive
    for that cluster, and the corresponding lineage hints are combined
    into a short descriptive note.

    This is a heuristic intended to make exploratory cluster output
    more interpretable -- it is explicitly NOT a validated clinical
    gating strategy and should be clearly labeled as such in the UI.

    Args:
        cluster_summary: Output of `build_cluster_summary`.
        marker_columns: Marker columns present in the summary (used to
            find the relevant `median_<marker>` columns).

    Returns:
        Copy of cluster_summary with an added `auto_gating_note` column.
    """
    annotated = cluster_summary.copy()
    notes = []

    available_lineage_markers = [m for m in _LINEAGE_MARKER_HINTS if m in marker_columns]

    # Use the median across clusters (weighted by cluster size) as a
    # rough proxy for "population-wide" typical expression per marker,
    # so that "high"/"low" is relative rather than absolute.
    overall_reference = {}
    for marker in available_lineage_markers:
        col = f"median_{marker}"
        if col in annotated.columns:
            overall_reference[marker] = float(
                np.average(annotated[col], weights=annotated["n_cells"])
            )

    for _, row in annotated.iterrows():
        if row["n_cells"] == 0:
            notes.append("Empty cluster.")
            continue
        if "cluster" in row and row["cluster"] == -1:
            notes.append("Noise / unassigned points (DBSCAN outliers, not a true population).")
            continue

        positive_markers = []
        for marker in available_lineage_markers:
            col = f"median_{marker}"
            if col not in row:
                continue
            ref = overall_reference.get(marker, 0)
            # A marker counts as "positive" for this cluster if its
            # median is noticeably above the weighted population
            # reference. The 1.25x threshold is a simple heuristic, not
            # a calibrated clinical cutoff.
            threshold = ref * 1.25 if ref > 0 else (row[col] * 0)
            if row[col] > threshold and row[col] > 0:
                positive_markers.append(marker)

        if not positive_markers:
            notes.append("No dominant lineage marker detected; possibly mixed/unclassified population.")
            continue

        hints = [_LINEAGE_MARKER_HINTS[m] for m in positive_markers if m != "CD45"]
        if not hints:
            notes.append("CD45+ leukocyte population; no specific lineage marker dominant.")
        else:
            note = "Likely " + " / ".join(sorted(set(hints))) + f" (positive for: {', '.join(positive_markers)})"
            notes.append(note)

    annotated["auto_gating_note"] = notes
    return annotated