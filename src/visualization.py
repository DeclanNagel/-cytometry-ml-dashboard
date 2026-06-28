"""
src/visualization.py
---------------------
Plotly-based visualization helpers for cytometry data exploration.

All functions return Plotly Figure objects so that the Streamlit app
can render them with `st.plotly_chart(fig, use_container_width=True)`
without any plotting logic living in app.py itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

try:
    import umap  # umap-learn

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when umap-learn is absent
    UMAP_AVAILABLE = False


class VisualizationError(ValueError):
    """Raised when visualization inputs are invalid."""


def scatter_plot_2d(
    df: pd.DataFrame,
    x_marker: str,
    y_marker: str,
    color_by: pd.Series | str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create an interactive 2D scatter plot of two markers.

    Args:
        df: DataFrame containing at least x_marker and y_marker columns.
        x_marker: Column name to plot on the X axis.
        y_marker: Column name to plot on the Y axis.
        color_by: Either a column name in df, a separate Series aligned
            with df's rows (e.g. cluster labels), or None for no coloring.
        title: Optional plot title. A sensible default is generated if omitted.

    Returns:
        Plotly Figure object.

    Raises:
        VisualizationError: If x_marker or y_marker are not in df.
    """
    if x_marker not in df.columns or y_marker not in df.columns:
        raise VisualizationError(f"Markers '{x_marker}' and/or '{y_marker}' not found in data.")

    plot_df = df.copy()
    color_col = None
    if color_by is not None:
        if isinstance(color_by, str):
            color_col = color_by
        else:
            plot_df = plot_df.copy()
            plot_df["_color"] = pd.Series(color_by).astype(str).values
            color_col = "_color"

    fig = px.scatter(
        plot_df,
        x=x_marker,
        y=y_marker,
        color=color_col,
        opacity=0.7,
        title=title or f"{y_marker} vs {x_marker}",
        labels={"_color": "Cluster"} if color_col == "_color" else {},
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text="Cluster" if color_col == "_color" else color_col)
    return fig


def marker_histogram(df: pd.DataFrame, marker: str, n_bins: int = 60) -> go.Figure:
    """Create a histogram of a single marker's expression distribution.

    Args:
        df: DataFrame containing the marker column.
        marker: Column name to plot.
        n_bins: Number of histogram bins.

    Returns:
        Plotly Figure object.

    Raises:
        VisualizationError: If marker is not in df.
    """
    if marker not in df.columns:
        raise VisualizationError(f"Marker '{marker}' not found in data.")

    fig = px.histogram(df, x=marker, nbins=n_bins, title=f"Distribution of {marker}")
    fig.update_layout(yaxis_title="Cell count", bargap=0.02)
    return fig


def compute_pca(
    df: pd.DataFrame, marker_columns: list[str], n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, PCA]:
    """Compute a PCA embedding of the marker columns.

    Args:
        df: Preprocessed (typically scaled) DataFrame.
        marker_columns: Columns to use as input features.
        n_components: Number of principal components to compute.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (embedding array of shape [n_rows, n_components], fitted PCA object).

    Raises:
        VisualizationError: If marker_columns is empty or n_components
            exceeds the number of available features.
    """
    if not marker_columns:
        raise VisualizationError("At least one marker column is required for PCA.")
    if n_components > len(marker_columns):
        raise VisualizationError("n_components cannot exceed the number of marker columns.")

    X = df[marker_columns].to_numpy()
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(X)
    return embedding, pca


def compute_umap(
    df: pd.DataFrame,
    marker_columns: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute a 2D UMAP embedding of the marker columns.

    Args:
        df: Preprocessed (typically scaled) DataFrame.
        marker_columns: Columns to use as input features.
        n_neighbors: UMAP n_neighbors hyperparameter (local vs. global structure tradeoff).
        min_dist: UMAP min_dist hyperparameter (how tightly points are packed).
        random_state: Seed for reproducibility.

    Returns:
        Embedding array of shape [n_rows, 2].

    Raises:
        VisualizationError: If umap-learn is not installed, or marker_columns is empty.
    """
    if not UMAP_AVAILABLE:
        raise VisualizationError(
            "umap-learn is not installed in this environment. "
            "Install it with `pip install umap-learn` to enable UMAP visualization."
        )
    if not marker_columns:
        raise VisualizationError("At least one marker column is required for UMAP.")

    X = df[marker_columns].to_numpy()
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))  # guard against tiny datasets
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def embedding_scatter(
    embedding: np.ndarray,
    color_by: pd.Series | np.ndarray | None,
    title: str,
    x_label: str = "Component 1",
    y_label: str = "Component 2",
) -> go.Figure:
    """Plot a 2D embedding (PCA or UMAP) as an interactive scatter plot.

    Args:
        embedding: Array of shape [n_rows, 2].
        color_by: Optional array/Series of labels (e.g. cluster assignment)
            used to color points. None for uncolored.
        title: Plot title.
        x_label: Label for the X axis.
        y_label: Label for the Y axis.

    Returns:
        Plotly Figure object.

    Raises:
        VisualizationError: If embedding does not have exactly 2 columns.
    """
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise VisualizationError("embedding must be a 2D array with exactly 2 columns.")

    plot_df = pd.DataFrame(embedding, columns=[x_label, y_label])
    color_col = None
    if color_by is not None:
        plot_df["Cluster"] = pd.Series(np.asarray(color_by)).astype(str).values
        color_col = "Cluster"

    fig = px.scatter(
        plot_df,
        x=x_label,
        y=y_label,
        color=color_col,
        opacity=0.7,
        title=title,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    return fig


def pca_explained_variance_plot(pca: PCA) -> go.Figure:
    """Create a bar chart of explained variance ratio per principal component.

    Args:
        pca: Fitted PCA object.

    Returns:
        Plotly Figure object.
    """
    n_components = len(pca.explained_variance_ratio_)
    labels = [f"PC{i+1}" for i in range(n_components)]
    fig = px.bar(
        x=labels,
        y=pca.explained_variance_ratio_ * 100,
        labels={"x": "Principal component", "y": "Explained variance (%)"},
        title="PCA explained variance",
    )
    return fig