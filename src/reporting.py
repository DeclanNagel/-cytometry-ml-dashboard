"""
src/reporting.py
-----------------
Reporting utilities: exporting cluster summaries and generating a
human-readable "Methods Summary" describing the preprocessing and
clustering steps applied during an analysis session.

Keeping this logic out of app.py makes it straightforward to unit
test report content and reuse it (e.g. in a future PDF/HTML export
feature) without touching any Streamlit-specific code.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def cluster_summary_to_csv_bytes(cluster_summary: pd.DataFrame) -> bytes:
    """Serialize a cluster summary DataFrame to CSV bytes for download.

    Args:
        cluster_summary: DataFrame produced by
            `clustering.build_cluster_summary` (optionally annotated by
            `clustering.interpret_clusters`).

    Returns:
        UTF-8 encoded CSV bytes, suitable for `st.download_button`.
    """
    return cluster_summary.to_csv(index=False).encode("utf-8")


def build_methods_summary(
    n_cells_raw: int,
    n_cells_final: int,
    marker_columns: list[str],
    preprocessing_steps: list[str],
    clustering_params: dict,
    n_clusters_found: int,
) -> str:
    """Generate a scientific-report-style "Methods" paragraph describing the analysis.

    Args:
        n_cells_raw: Number of cells (rows) in the originally uploaded/generated dataset.
        n_cells_final: Number of cells remaining after preprocessing.
        marker_columns: Marker columns used in the analysis.
        preprocessing_steps: Ordered list of preprocessing step descriptions
            (as produced by `preprocessing.run_preprocessing_pipeline`).
        clustering_params: Dict of clustering parameters actually used
            (as produced by `clustering.run_clustering`).
        n_clusters_found: Number of distinct clusters identified
            (excluding DBSCAN noise, where applicable).

    Returns:
        A multi-paragraph markdown-formatted string summarizing the
        full analysis pipeline in the style of a methods section.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    marker_list_str = ", ".join(marker_columns)
    steps_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(preprocessing_steps))

    algo = clustering_params.get("algorithm", "Unknown")
    param_items = {k: v for k, v in clustering_params.items() if k != "algorithm"}
    param_str = ", ".join(f"{k}={v}" for k, v in param_items.items())

    summary = f"""
**Analysis date:** {timestamp}

**Dataset:** {n_cells_raw} cells acquired; {n_cells_final} cells retained for
downstream analysis after quality control. {len(marker_columns)} marker
channels were analyzed: {marker_list_str}.

**Preprocessing pipeline:**

{steps_str}

**Clustering / automated gating:** Unsupervised clustering was performed
using **{algo}** ({param_str if param_str else "default parameters"}),
identifying **{n_clusters_found}** distinct cluster(s) across the marker
panel. Cluster identity was annotated using a rule-based heuristic that
compares each cluster's median marker expression to the population-wide
weighted median, flagging markers at least 1.25x the reference level as
"positive" for that cluster and matching the resulting marker signature
to canonical immune lineage markers (CD3 = T cell, CD4 = helper T cell,
CD8 = cytotoxic T cell, CD19 = B cell, CD14 = monocyte, CD56 = NK cell,
CD45 = pan-leukocyte).

**Dimensionality reduction:** Principal Component Analysis (PCA) and,
where available, UMAP were used to project the high-dimensional marker
space into two dimensions for visual inspection of cluster separation
and population structure.

**Caveats:** Cluster-to-population assignments produced by the
auto-gating heuristic are exploratory and intended to accelerate
manual review; they are not a validated diagnostic or clinical gating
strategy and should be confirmed against established gating panels and
expert review before any biological or clinical interpretation.
""".strip()

    return summary


def build_preprocessing_steps_markdown(steps: list[str]) -> str:
    """Format a list of preprocessing step strings as a markdown bullet list.

    Args:
        steps: List of human-readable preprocessing step descriptions.

    Returns:
        Markdown-formatted bullet list string.
    """
    return "\n".join(f"- {step}" for step in steps)


def build_clustering_params_markdown(params: dict) -> str:
    """Format a clustering parameters dict as a markdown bullet list.

    Args:
        params: Dict of parameter name -> value (as returned by
            `clustering.run_clustering`).

    Returns:
        Markdown-formatted bullet list string.
    """
    return "\n".join(f"- **{key}**: {value}" for key, value in params.items())