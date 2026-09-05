"""Authoritative fidelity and structural-validity metric kernels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance


SUMMARY_STATS = ("first", "last", "min", "max", "median")


def normalized_wasserstein(real: pd.Series, synthetic: pd.Series, seed: int = 42) -> float:
    """Seed-level normalized Wasserstein metric used in the 15-seed analysis."""
    r = real.dropna().astype(float).to_numpy()
    s = synthetic.dropna().astype(float).to_numpy()
    rng = np.random.RandomState(seed)
    r = rng.choice(r, min(5000, len(r)), replace=False)
    s = rng.choice(s, min(5000, len(s)), replace=False)
    return float(wasserstein_distance(r, s) / real.dropna().astype(float).std())


def correlation_rank_preservation(real_medians: pd.DataFrame, synthetic_medians: pd.DataFrame) -> float:
    real = real_medians.corr(method="pearson").to_numpy()
    syn = synthetic_medians.corr(method="pearson").to_numpy()
    upper = np.triu_indices(real.shape[0], 1)
    return float(spearmanr(syn[upper], real[upper]).statistic)


def per_variable_structural_violation(df: pd.DataFrame, variable: str) -> pd.Series:
    first, last, minimum, maximum, median = (
        df[f"{variable}_{suffix}"] for suffix in SUMMARY_STATS
    )
    return ((minimum > median) | (median > maximum) | (minimum > maximum)
            | (first < minimum) | (first > maximum)
            | (last < minimum) | (last > maximum)).fillna(False)


def structural_rates(df: pd.DataFrame, variables: list[str]) -> dict[str, float]:
    matrix = np.column_stack([per_variable_structural_violation(df, v) for v in variables])
    return {
        "variable_row_violation_rate": float(matrix.mean()),
        "any_row_violation_rate": float(matrix.any(axis=1).mean()),
    }


def measurement_coupling(df: pd.DataFrame, variables: list[str]) -> dict[str, float]:
    lactate = float(df["lactate_median"].corr(df["lactate_observed_hour_density"]))
    vitals = ["heart_rate", "sbp", "dbp", "respiratory_rate", "spo2"]
    z = pd.DataFrame({v: (df[f"{v}_median"] - df[f"{v}_median"].mean()) /
                         df[f"{v}_median"].std() for v in vitals})
    density = df[[f"{v}_observed_hour_density" for v in variables]].mean(axis=1)
    abnormality = float(z.abs().mean(axis=1).corr(density))
    return {"lactate_value_density": lactate, "abnormality_density": abnormality}
