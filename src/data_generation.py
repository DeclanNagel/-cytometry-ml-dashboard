"""
src/data_generation.py
-----------------------
Synthetic flow-cytometry-like data generator.

This module exists so that the dashboard is immediately usable without
requiring a real dataset upload. It simulates a peripheral blood
mononuclear cell (PBMC) sample with several well-known immune cell
populations, each defined by a characteristic marker expression
signature (high/low combinations of CD3, CD4, CD8, CD19, CD14, CD56)
plus light-scatter channels (FSC-A, SSC-A) that loosely track cell
size/granularity.

The populations simulated are simplified analogues of real biology and
are intended for demonstration/portfolio purposes only -- they are NOT
derived from real patient data and should not be used for any clinical
or research inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Marker panel used throughout the app. Order matters for downstream
# default UI choices (e.g. default X/Y axis selections).
MARKER_COLUMNS = [
    "FSC-A",
    "SSC-A",
    "CD3",
    "CD4",
    "CD8",
    "CD19",
    "CD14",
    "CD56",
    "CD45",
]

# Population definitions: each population has a name, an approximate
# proportion of the total population, and a mean expression vector for
# every marker in MARKER_COLUMNS (on an arbitrary fluorescence-intensity
# scale, roughly 0-20,000 to resemble raw FCS scale before transform).
# Standard deviations are supplied per-marker to control spread/overlap.
_POPULATIONS = {
    "CD4+ T cells": {
        "proportion": 0.30,
        "mean": {
            "FSC-A": 55000, "SSC-A": 20000,
            "CD3": 14000, "CD4": 12000, "CD8": 300,
            "CD19": 200, "CD14": 150, "CD56": 200, "CD45": 15000,
        },
        "std": {
            "FSC-A": 6000, "SSC-A": 3000,
            "CD3": 2500, "CD4": 2200, "CD8": 250,
            "CD19": 150, "CD14": 150, "CD56": 200, "CD45": 2000,
        },
    },
    "CD8+ T cells": {
        "proportion": 0.20,
        "mean": {
            "FSC-A": 53000, "SSC-A": 19000,
            "CD3": 13500, "CD4": 250, "CD8": 11500,
            "CD19": 200, "CD14": 150, "CD56": 250, "CD45": 14500,
        },
        "std": {
            "FSC-A": 6000, "SSC-A": 3000,
            "CD3": 2400, "CD4": 200, "CD8": 2100,
            "CD19": 150, "CD14": 150, "CD56": 200, "CD45": 2000,
        },
    },
    "B cells": {
        "proportion": 0.15,
        "mean": {
            "FSC-A": 50000, "SSC-A": 18000,
            "CD3": 200, "CD4": 150, "CD8": 150,
            "CD19": 13000, "CD14": 150, "CD56": 200, "CD45": 16000,
        },
        "std": {
            "FSC-A": 5500, "SSC-A": 2800,
            "CD3": 150, "CD4": 150, "CD8": 150,
            "CD19": 2300, "CD14": 150, "CD56": 200, "CD45": 2200,
        },
    },
    "Monocytes": {
        "proportion": 0.20,
        "mean": {
            "FSC-A": 75000, "SSC-A": 45000,
            "CD3": 200, "CD4": 1500, "CD8": 150,
            "CD19": 150, "CD14": 14000, "CD56": 200, "CD45": 13000,
        },
        "std": {
            "FSC-A": 8000, "SSC-A": 6000,
            "CD3": 150, "CD4": 600, "CD8": 150,
            "CD19": 150, "CD14": 2500, "CD56": 200, "CD45": 2000,
        },
    },
    "NK cells": {
        "proportion": 0.10,
        "mean": {
            "FSC-A": 52000, "SSC-A": 22000,
            "CD3": 250, "CD4": 200, "CD8": 800,
            "CD19": 150, "CD14": 150, "CD56": 12000, "CD45": 14000,
        },
        "std": {
            "FSC-A": 6000, "SSC-A": 3500,
            "CD3": 150, "CD4": 150, "CD8": 400,
            "CD19": 150, "CD14": 150, "CD56": 2200, "CD45": 2000,
        },
    },
    "Debris/Dead cells": {
        "proportion": 0.05,
        "mean": {
            "FSC-A": 8000, "SSC-A": 30000,
            "CD3": 300, "CD4": 300, "CD8": 300,
            "CD19": 300, "CD14": 300, "CD56": 300, "CD45": 2000,
        },
        "std": {
            "FSC-A": 4000, "SSC-A": 12000,
            "CD3": 300, "CD4": 300, "CD8": 300,
            "CD19": 300, "CD14": 300, "CD56": 300, "CD45": 1500,
        },
    },
}


def generate_synthetic_cytometry_data(
    n_cells: int = 5000,
    random_state: int | None = 42,
    include_population_labels: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic flow-cytometry-like dataset.

    Simulates a PBMC sample composed of several immune cell populations
    (CD4+ T cells, CD8+ T cells, B cells, Monocytes, NK cells, and a
    small debris/dead-cell fraction). Each population is drawn from a
    multivariate Gaussian (per-marker independent) parameterized by
    biologically-plausible mean/SD expression for each marker.

    Args:
        n_cells: Total number of cells (rows) to simulate.
        random_state: Seed for reproducibility. Use ``None`` for a
            non-deterministic draw.
        include_population_labels: If True, includes a
            ``true_population`` ground-truth column. This is useful for
            validating clustering quality in a demo context, but is
            clearly labeled as synthetic ground truth and is not a
            "marker" column used in analysis.

    Returns:
        DataFrame with one row per simulated cell, columns for every
        marker in ``MARKER_COLUMNS``, and optionally a
        ``true_population`` label column.

    Raises:
        ValueError: If n_cells is not a positive integer.
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be a positive integer.")

    rng = np.random.default_rng(random_state)

    pop_names = list(_POPULATIONS.keys())
    proportions = np.array([_POPULATIONS[p]["proportion"] for p in pop_names])
    proportions = proportions / proportions.sum()  # normalize defensively

    # Assign each cell to a population according to the proportions.
    pop_assignment = rng.choice(pop_names, size=n_cells, p=proportions)

    data = {marker: np.empty(n_cells) for marker in MARKER_COLUMNS}

    for pop_name in pop_names:
        mask = pop_assignment == pop_name
        n_in_pop = int(mask.sum())
        if n_in_pop == 0:
            continue
        means = _POPULATIONS[pop_name]["mean"]
        stds = _POPULATIONS[pop_name]["std"]
        for marker in MARKER_COLUMNS:
            draws = rng.normal(loc=means[marker], scale=stds[marker], size=n_in_pop)
            data[marker][mask] = draws

    # Clip to a realistic non-negative fluorescence/scatter range and
    # add a small amount of detector noise.
    df = pd.DataFrame(data)
    noise = rng.normal(loc=0, scale=50, size=df.shape)
    df = df + noise
    df[MARKER_COLUMNS] = df[MARKER_COLUMNS].clip(lower=0)

    if include_population_labels:
        df["true_population"] = pop_assignment

    # Shuffle rows so populations aren't trivially ordered.
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df


def get_population_names() -> list[str]:
    """Return the list of simulated population names (for display/legend use)."""
    return list(_POPULATIONS.keys())