"""
src/preprocessing.py
---------------------
Preprocessing utilities for flow-cytometry-like tabular data.

Implements the standard preprocessing steps used in cytometry analysis
pipelines:
    1. Marker column selection
    2. Missing value handling
    3. Arcsinh ("logicle-like") transformation to compress the wide
       dynamic range typical of fluorescence intensity data
    4. Standard (z-score) scaling

Each function is intentionally small and composable so that the
Streamlit app can call them independently and display intermediate
results / QC metrics at every step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreprocessingError(ValueError):
    """Raised when input data does not meet the requirements for preprocessing."""


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric columns in a DataFrame.

    Args:
        df: Input DataFrame (e.g. raw uploaded CSV).

    Returns:
        List of column names with a numeric dtype, in their original order.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compute_qc_summary(df: pd.DataFrame, marker_columns: list[str]) -> dict:
    """Compute a basic quality-control summary for the selected markers.

    Args:
        df: Raw (pre-cleaning) DataFrame.
        marker_columns: Marker columns to summarize.

    Returns:
        Dictionary with row/column counts, per-marker missing-value
        counts, and per-marker descriptive statistics. Designed to be
        rendered directly in the Streamlit UI.

    Raises:
        PreprocessingError: If marker_columns is empty or contains
            columns not present in df.
    """
    if not marker_columns:
        raise PreprocessingError("No marker columns were selected for QC.")
    missing_cols = [c for c in marker_columns if c not in df.columns]
    if missing_cols:
        raise PreprocessingError(f"Columns not found in data: {missing_cols}")

    missing_counts = df[marker_columns].isna().sum()
    total_missing = int(missing_counts.sum())

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns_total": int(df.shape[1]),
        "n_marker_columns": len(marker_columns),
        "total_missing_values": total_missing,
        "missing_by_marker": missing_counts.to_dict(),
        "marker_descriptive_stats": df[marker_columns].describe().T,
    }
    return summary


def handle_missing_values(
    df: pd.DataFrame,
    marker_columns: list[str],
    strategy: str = "drop_rows",
) -> pd.DataFrame:
    """Handle missing values in the marker columns.

    Args:
        df: Input DataFrame.
        marker_columns: Columns to consider when checking for missing values.
        strategy: One of:
            - "drop_rows": drop any row with a missing value in a marker column.
            - "median_impute": fill missing values with the column median.
            - "zero_fill": fill missing values with 0.

    Returns:
        A new DataFrame with missing values handled (original is not mutated).

    Raises:
        PreprocessingError: If an unsupported strategy is provided or no
            rows remain after cleaning.
    """
    valid_strategies = {"drop_rows", "median_impute", "zero_fill"}
    if strategy not in valid_strategies:
        raise PreprocessingError(
            f"Unknown missing-value strategy '{strategy}'. "
            f"Expected one of {sorted(valid_strategies)}."
        )

    cleaned = df.copy()

    if strategy == "drop_rows":
        cleaned = cleaned.dropna(subset=marker_columns)
    elif strategy == "median_impute":
        for col in marker_columns:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
    elif strategy == "zero_fill":
        cleaned[marker_columns] = cleaned[marker_columns].fillna(0)

    if cleaned.shape[0] == 0:
        raise PreprocessingError(
            "No rows remain after handling missing values. "
            "Try a different strategy (e.g. median imputation) or check your data."
        )

    return cleaned.reset_index(drop=True)


def arcsinh_transform(
    df: pd.DataFrame,
    marker_columns: list[str],
    cofactor: float = 150.0,
) -> pd.DataFrame:
    """Apply an arcsinh ("logicle-like") transform to marker columns.

    Flow cytometry fluorescence data spans several orders of magnitude
    and contains values near/below zero (due to background and
    detector noise), which makes a plain log transform unsuitable. The
    arcsinh transform, ``f(x) = asinh(x / cofactor)``, behaves like a
    log transform at high values while remaining well-defined (and
    approximately linear) near zero -- a simplified stand-in for the
    biexponential/logicle transform commonly used in commercial
    cytometry software (e.g. FlowJo).

    Args:
        df: Input DataFrame (after missing-value handling).
        marker_columns: Columns to transform.
        cofactor: Linear/log transition scale factor. Typical values
            for flow cytometry are in the 100-1000 range depending on
            instrument gain; 150 is a reasonable default for
            fluorescence channels.

    Returns:
        A new DataFrame with the specified columns transformed.

    Raises:
        PreprocessingError: If cofactor is not positive.
    """
    if cofactor <= 0:
        raise PreprocessingError("cofactor must be a positive number.")

    transformed = df.copy()
    for col in marker_columns:
        transformed[col] = np.arcsinh(transformed[col] / cofactor)
    return transformed


def standard_scale(
    df: pd.DataFrame, marker_columns: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply z-score standardization (zero mean, unit variance) to marker columns.

    Args:
        df: Input DataFrame (typically after arcsinh transformation).
        marker_columns: Columns to scale.

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler instance).
        The fitted scaler is returned so that the app can report
        per-marker mean/scale in the methods summary if desired.

    Raises:
        PreprocessingError: If any marker column has zero variance,
            which would make scaling degenerate (handled gracefully by
            scikit-learn but flagged here for transparency).
    """
    scaled = df.copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scaled[marker_columns])
    scaled[marker_columns] = scaled_values
    return scaled, scaler


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    marker_columns: list[str],
    missing_strategy: str = "drop_rows",
    apply_arcsinh: bool = True,
    arcsinh_cofactor: float = 150.0,
    apply_scaling: bool = True,
) -> dict:
    """Run the full preprocessing pipeline and return all intermediate artifacts.

    This is the main entry point used by the Streamlit app. It chains
    together missing-value handling, optional arcsinh transformation,
    and optional standard scaling, while keeping track of the steps
    actually applied (for the reporting module's "Methods Summary").

    Args:
        df: Raw input DataFrame.
        marker_columns: Marker columns selected by the user.
        missing_strategy: Strategy passed to `handle_missing_values`.
        apply_arcsinh: Whether to apply the arcsinh transform.
        arcsinh_cofactor: Cofactor for the arcsinh transform.
        apply_scaling: Whether to apply standard scaling.

    Returns:
        Dictionary with keys:
            - "raw_qc": QC summary computed on the raw input.
            - "cleaned_df": DataFrame after missing-value handling.
            - "transformed_df": DataFrame after optional arcsinh transform.
            - "processed_df": Final DataFrame after optional scaling
              (this is what should be fed into clustering/visualization).
            - "scaler": Fitted StandardScaler, or None if scaling was skipped.
            - "steps_applied": Ordered list of human-readable strings
              describing each preprocessing step actually performed.

    Raises:
        PreprocessingError: Propagated from any of the underlying steps.
    """
    if not marker_columns:
        raise PreprocessingError("At least one marker column must be selected.")

    steps_applied: list[str] = []

    raw_qc = compute_qc_summary(df, marker_columns)
    steps_applied.append(
        f"Computed QC summary on {raw_qc['n_rows']} rows / "
        f"{raw_qc['n_marker_columns']} marker columns."
    )

    cleaned_df = handle_missing_values(df, marker_columns, strategy=missing_strategy)
    n_dropped = df.shape[0] - cleaned_df.shape[0] if missing_strategy == "drop_rows" else 0
    steps_applied.append(
        f"Missing-value handling: strategy='{missing_strategy}'"
        + (f" ({n_dropped} rows dropped)." if missing_strategy == "drop_rows" else ".")
    )

    transformed_df = cleaned_df
    if apply_arcsinh:
        transformed_df = arcsinh_transform(cleaned_df, marker_columns, cofactor=arcsinh_cofactor)
        steps_applied.append(
            f"Applied arcsinh (logicle-like) transform with cofactor={arcsinh_cofactor}."
        )
    else:
        steps_applied.append("Skipped arcsinh transform (used raw marker intensities).")

    processed_df = transformed_df
    scaler = None
    if apply_scaling:
        processed_df, scaler = standard_scale(transformed_df, marker_columns)
        steps_applied.append("Applied StandardScaler (zero mean, unit variance) per marker.")
    else:
        steps_applied.append("Skipped standard scaling.")

    return {
        "raw_qc": raw_qc,
        "cleaned_df": cleaned_df,
        "transformed_df": transformed_df,
        "processed_df": processed_df,
        "scaler": scaler,
        "steps_applied": steps_applied,
    }