"""
app.py
------
Cytometry ML Analysis Dashboard

An interactive Streamlit application for exploring flow-cytometry-like
single-cell data: upload or simulate data, preprocess it, visualize
marker distributions and dimensionality-reduced embeddings, run
ML-based clustering ("automated gating"), and export a cluster summary
report with an auto-generated Methods section.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from src.clustering import (
    ClusteringError,
    build_cluster_summary,
    interpret_clusters,
    run_clustering,
)
from src.data_generation import MARKER_COLUMNS, generate_synthetic_cytometry_data
from src.preprocessing import (
    PreprocessingError,
    get_numeric_columns,
    run_preprocessing_pipeline,
)
from src.reporting import (
    build_clustering_params_markdown,
    build_methods_summary,
    build_preprocessing_steps_markdown,
    cluster_summary_to_csv_bytes,
)
from src.visualization import (
    UMAP_AVAILABLE,
    VisualizationError,
    compute_pca,
    compute_umap,
    embedding_scatter,
    marker_histogram,
    pca_explained_variance_plot,
    scatter_plot_2d,
)

st.set_page_config(
    page_title="Cytometry ML Analysis Dashboard",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialize keys used in st.session_state if they don't already exist."""
    defaults = {
        "raw_df": None,
        "data_source": None,  # "uploaded" or "synthetic"
        "preprocessing_result": None,
        "marker_columns": [],
        "cluster_labels": None,
        "cluster_params": None,
        "cluster_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: data loading
# ---------------------------------------------------------------------------
st.sidebar.title("Cytometry ML Dashboard")
st.sidebar.markdown("Upload data or generate a synthetic sample to get started.")

with st.sidebar.expander("📄 Expected CSV format", expanded=False):
    st.markdown(
        """
        Your CSV should contain one row per cell and one column per
        marker/channel, with **numeric** values only in marker columns
        (non-numeric columns, e.g. an ID column, are simply ignored
        during marker selection).

        Example columns: `FSC-A`, `SSC-A`, `CD3`, `CD4`, `CD8`, `CD19`,
        `CD14`, `CD56`, `CD45`. Any marker/channel names are supported
        — the app lets you choose which columns to treat as markers
        after upload.
        """
    )

data_mode = st.sidebar.radio(
    "Data source",
    options=["Upload CSV", "Generate synthetic sample"],
    index=1,
)

if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a cytometry CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            if new_df.empty:
                st.sidebar.error("The uploaded CSV is empty.")
            else:
                st.session_state.raw_df = new_df
                st.session_state.data_source = "uploaded"
                # Reset downstream state when new data is loaded.
                st.session_state.preprocessing_result = None
                st.session_state.cluster_labels = None
                st.session_state.cluster_summary = None
        except Exception as exc:  # noqa: BLE001 - surfaced to user, not swallowed
            st.sidebar.error(f"Could not read CSV file: {exc}")
else:
    st.sidebar.markdown("**Synthetic PBMC-like sample generator**")
    n_cells = st.sidebar.slider("Number of cells", min_value=500, max_value=20000, value=5000, step=500)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    if st.sidebar.button("Generate synthetic dataset", type="primary"):
        try:
            st.session_state.raw_df = generate_synthetic_cytometry_data(
                n_cells=n_cells, random_state=int(seed)
            )
            st.session_state.data_source = "synthetic"
            st.session_state.preprocessing_result = None
            st.session_state.cluster_labels = None
            st.session_state.cluster_summary = None
        except ValueError as exc:
            st.sidebar.error(str(exc))

    # Auto-generate a default sample on first load so the app is
    # immediately usable without any clicks.
    if st.session_state.raw_df is None:
        st.session_state.raw_df = generate_synthetic_cytometry_data(n_cells=5000, random_state=42)
        st.session_state.data_source = "synthetic"


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Cytometry ML Analysis Dashboard")
st.caption(
    "Upload flow-cytometry-like data, preprocess it, explore cell populations, "
    "and run ML-based clustering to generate automated gating summaries."
)

if st.session_state.raw_df is None:
    st.info("Upload a CSV file or generate a synthetic dataset from the sidebar to begin.")
    st.stop()

raw_df = st.session_state.raw_df

source_label = "Uploaded file" if st.session_state.data_source == "uploaded" else "Synthetic sample"
st.success(f"{source_label} loaded: **{raw_df.shape[0]:,}** rows × **{raw_df.shape[1]}** columns")

with st.expander("Preview raw data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

tab_preprocess, tab_visualize, tab_cluster, tab_report = st.tabs(
    ["1Preprocessing", "2Visualization", "Clustering / Auto-gating", " Report"]
)


# ---------------------------------------------------------------------------
# Tab 1: Preprocessing
# ---------------------------------------------------------------------------
with tab_preprocess:
    st.header("Preprocessing & quality control")

    numeric_cols = get_numeric_columns(raw_df)
    if not numeric_cols:
        st.error(
            "No numeric columns were found in this dataset. "
            "Please upload a CSV with numeric marker columns."
        )
        st.stop()

    default_markers = [c for c in numeric_cols if c in MARKER_COLUMNS] or numeric_cols
    marker_columns = st.multiselect(
        "Select marker columns to use in the analysis",
        options=numeric_cols,
        default=default_markers,
        help="Only numeric columns can be selected as markers.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        missing_strategy = st.selectbox(
            "Missing value strategy",
            options=["drop_rows", "median_impute", "zero_fill"],
            format_func=lambda s: {
                "drop_rows": "Drop rows with missing values",
                "median_impute": "Impute with column median",
                "zero_fill": "Fill with zero",
            }[s],
        )
        apply_arcsinh = st.checkbox("Apply arcsinh (logicle-like) transform", value=True)
        arcsinh_cofactor = st.number_input(
            "Arcsinh cofactor", min_value=1.0, max_value=10000.0, value=150.0, step=10.0,
            disabled=not apply_arcsinh,
        )
    with col_b:
        apply_scaling = st.checkbox("Apply standard scaling (z-score)", value=True)
        st.markdown(
            "**Standard scaling** centers each marker to zero mean and unit "
            "variance, which is recommended before distance-based clustering "
            "algorithms such as KMeans and DBSCAN."
        )

    run_preprocess = st.button("Run preprocessing", type="primary")

    if run_preprocess or st.session_state.preprocessing_result is not None:
        try:
            if run_preprocess:
                if not marker_columns:
                    st.error("Please select at least one marker column.")
                    st.stop()
                st.session_state.preprocessing_result = run_preprocessing_pipeline(
                    raw_df,
                    marker_columns,
                    missing_strategy=missing_strategy,
                    apply_arcsinh=apply_arcsinh,
                    arcsinh_cofactor=arcsinh_cofactor,
                    apply_scaling=apply_scaling,
                )
                st.session_state.marker_columns = marker_columns
                # Invalidate downstream clustering results on re-run.
                st.session_state.cluster_labels = None
                st.session_state.cluster_summary = None

            result = st.session_state.preprocessing_result
            qc = result["raw_qc"]

            st.subheader("QC summary")
            qc_col1, qc_col2, qc_col3, qc_col4 = st.columns(4)
            qc_col1.metric("Rows (cells)", f"{qc['n_rows']:,}")
            qc_col2.metric("Total columns", qc["n_columns_total"])
            qc_col3.metric("Marker columns", qc["n_marker_columns"])
            qc_col4.metric("Missing values", qc["total_missing_values"])

            st.markdown("**Per-marker missing value counts:**")
            missing_series = pd.Series(qc["missing_by_marker"], name="missing_count")
            st.dataframe(missing_series.to_frame(), use_container_width=True)

            st.markdown("**Marker descriptive statistics (raw data):**")
            st.dataframe(qc["marker_descriptive_stats"], use_container_width=True)

            st.subheader("Steps applied")
            st.markdown(build_preprocessing_steps_markdown(result["steps_applied"]))

            st.subheader("Processed data preview")
            st.dataframe(result["processed_df"].head(20), use_container_width=True)

        except PreprocessingError as exc:
            st.error(f"Preprocessing failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"An unexpected error occurred during preprocessing: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
    else:
        st.info("Configure preprocessing options above and click **Run preprocessing** to continue.")


# ---------------------------------------------------------------------------
# Tab 2: Visualization
# ---------------------------------------------------------------------------
with tab_visualize:
    st.header("Cell population visualization")

    if st.session_state.preprocessing_result is None:
        st.warning("Please run preprocessing first (see the **Preprocessing** tab).")
    else:
        processed_df = st.session_state.preprocessing_result["processed_df"]
        markers = st.session_state.marker_columns
        cluster_labels = st.session_state.cluster_labels  # may be None

        st.subheader("2D marker scatter plot")
        scatter_col1, scatter_col2 = st.columns(2)
        with scatter_col1:
            x_marker = st.selectbox("X-axis marker", options=markers, index=0, key="scatter_x")
        with scatter_col2:
            default_y_idx = 1 if len(markers) > 1 else 0
            y_marker = st.selectbox("Y-axis marker", options=markers, index=default_y_idx, key="scatter_y")

        color_option = st.radio(
            "Color points by",
            options=["None", "Cluster (if available)"],
            horizontal=True,
        )
        try:
            color_by = cluster_labels if (color_option == "Cluster (if available)" and cluster_labels is not None) else None
            if color_option == "Cluster (if available)" and cluster_labels is None:
                st.info("No cluster labels yet — run clustering in the next tab to color by cluster.")
            fig_scatter = scatter_plot_2d(processed_df, x_marker, y_marker, color_by=color_by)
            st.plotly_chart(fig_scatter, use_container_width=True)
        except VisualizationError as exc:
            st.error(str(exc))

        st.subheader("Marker distribution histogram")
        hist_marker = st.selectbox("Marker", options=markers, key="hist_marker")
        try:
            fig_hist = marker_histogram(processed_df, hist_marker)
            st.plotly_chart(fig_hist, use_container_width=True)
        except VisualizationError as exc:
            st.error(str(exc))

        st.subheader("Dimensionality reduction")
        dr_method = st.radio("Method", options=["PCA", "UMAP"], horizontal=True)

        if dr_method == "PCA":
            try:
                embedding, pca_obj = compute_pca(processed_df, markers, n_components=2)
                fig_pca = embedding_scatter(
                    embedding, cluster_labels, title="PCA projection",
                    x_label="PC1", y_label="PC2",
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                st.plotly_chart(pca_explained_variance_plot(pca_obj), use_container_width=True)
            except VisualizationError as exc:
                st.error(str(exc))
        else:
            if not UMAP_AVAILABLE:
                st.warning(
                    "umap-learn is not installed in this environment. "
                    "Install it with `pip install umap-learn` to enable UMAP visualization."
                )
            else:
                umap_col1, umap_col2 = st.columns(2)
                with umap_col1:
                    n_neighbors = st.slider("n_neighbors", min_value=2, max_value=100, value=15)
                with umap_col2:
                    min_dist = st.slider("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                if st.button("Compute UMAP embedding"):
                    with st.spinner("Computing UMAP embedding..."):
                        try:
                            umap_embedding = compute_umap(
                                processed_df, markers, n_neighbors=n_neighbors, min_dist=min_dist
                            )
                            st.session_state["_umap_embedding"] = umap_embedding
                        except VisualizationError as exc:
                            st.error(str(exc))
                if "_umap_embedding" in st.session_state:
                    fig_umap = embedding_scatter(
                        st.session_state["_umap_embedding"], cluster_labels,
                        title="UMAP projection", x_label="UMAP-1", y_label="UMAP-2",
                    )
                    st.plotly_chart(fig_umap, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Clustering / automated gating
# ---------------------------------------------------------------------------
with tab_cluster:
    st.header("ML-based clustering & automated gating")

    if st.session_state.preprocessing_result is None:
        st.warning("Please run preprocessing first (see the **Preprocessing** tab).")
    else:
        processed_df = st.session_state.preprocessing_result["processed_df"]
        cleaned_raw_df = st.session_state.preprocessing_result["cleaned_df"]
        markers = st.session_state.marker_columns

        algorithm = st.selectbox(
            "Clustering algorithm", options=["KMeans", "DBSCAN", "Gaussian Mixture"]
        )

        params: dict = {}
        if algorithm == "KMeans":
            params["n_clusters"] = st.slider("n_clusters", min_value=2, max_value=20, value=6)
            params["random_state"] = st.number_input("random_state", min_value=0, max_value=10_000, value=42)
        elif algorithm == "DBSCAN":
            params["eps"] = st.slider("eps", min_value=0.05, max_value=5.0, value=0.8, step=0.05)
            params["min_samples"] = st.slider("min_samples", min_value=2, max_value=100, value=15)
            st.caption(
                "DBSCAN does not require specifying the number of clusters in advance, "
                "but is sensitive to the `eps` and `min_samples` parameters. Points "
                "labeled cluster `-1` are considered noise/outliers."
            )
        else:  # Gaussian Mixture
            params["n_components"] = st.slider("n_components", min_value=2, max_value=20, value=6)
            params["random_state"] = st.number_input("random_state", min_value=0, max_value=10_000, value=42)

        if st.button("Run clustering", type="primary"):
            try:
                labels, params_used = run_clustering(processed_df, markers, algorithm, **params)
                st.session_state.cluster_labels = labels
                st.session_state.cluster_params = params_used

                summary = build_cluster_summary(
                    processed_df, markers, labels, raw_marker_df=cleaned_raw_df
                )
                annotated_summary = interpret_clusters(summary, markers)
                st.session_state.cluster_summary = annotated_summary
            except ClusteringError as exc:
                st.error(f"Clustering failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"An unexpected error occurred during clustering: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

        if st.session_state.cluster_summary is not None:
            st.subheader("Cluster summary")
            st.caption(
                "Median marker expression is reported on the **original (untransformed) scale** "
                "for interpretability, even though clustering was performed on preprocessed data."
            )
            st.dataframe(st.session_state.cluster_summary, use_container_width=True)

            n_clusters_found = int(
                (st.session_state.cluster_summary["cluster"] != -1).sum()
            )
            n_noise = int((st.session_state.cluster_summary["cluster"] == -1).sum())
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Clusters identified", n_clusters_found)
            metric_col2.metric("Noise cluster present", "Yes" if n_noise else "No")

            st.subheader("Cluster visualization (first two selected markers)")
            try:
                fig = scatter_plot_2d(
                    processed_df, markers[0], markers[1] if len(markers) > 1 else markers[0],
                    color_by=st.session_state.cluster_labels,
                    title="Clusters colored on marker scatter",
                )
                st.plotly_chart(fig, use_container_width=True)
            except VisualizationError as exc:
                st.error(str(exc))
        else:
            st.info("Configure parameters above and click **Run clustering** to identify cell populations.")


# ---------------------------------------------------------------------------
# Tab 4: Report
# ---------------------------------------------------------------------------
with tab_report:
    st.header("Analysis report")

    if st.session_state.cluster_summary is None:
        st.warning("Please run preprocessing and clustering first to generate a report.")
    else:
        result = st.session_state.preprocessing_result
        markers = st.session_state.marker_columns
        cluster_summary = st.session_state.cluster_summary
        cluster_params = st.session_state.cluster_params

        st.subheader("Downloadable cluster summary")
        csv_bytes = cluster_summary_to_csv_bytes(cluster_summary)
        st.download_button(
            label="⬇️ Download cluster summary (CSV)",
            data=csv_bytes,
            file_name="cluster_summary.csv",
            mime="text/csv",
        )
        st.dataframe(cluster_summary, use_container_width=True)

        st.subheader("Preprocessing steps used")
        st.markdown(build_preprocessing_steps_markdown(result["steps_applied"]))

        st.subheader("Clustering model parameters used")
        st.markdown(build_clustering_params_markdown(cluster_params))

        st.subheader("📋 Methods Summary")
        n_clusters_found = int((cluster_summary["cluster"] != -1).sum())
        methods_text = build_methods_summary(
            n_cells_raw=raw_df.shape[0],
            n_cells_final=result["processed_df"].shape[0],
            marker_columns=markers,
            preprocessing_steps=result["steps_applied"],
            clustering_params=cluster_params,
            n_clusters_found=n_clusters_found,
        )
        st.markdown(methods_text)

        st.download_button(
            label="⬇️ Download Methods Summary (Markdown)",
            data=methods_text.encode("utf-8"),
            file_name="methods_summary.md",
            mime="text/markdown",
        )


st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit, scikit-learn, Plotly, and UMAP. "
    "Synthetic data is simulated for demonstration purposes only."
)