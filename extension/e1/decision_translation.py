#!/usr/bin/env python3
"""Public, data-agnostic E1 decision-translation helpers.

These functions do not load clinical records, patient predictions, fitted clinical
models, hospital identifiers, bootstrap indices, or private covariance assets.
They translate already estimated suitability probabilities into the frozen P95
three-state decision and aggregate realization-level decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def p95_state(probability: float, confidence: float = 0.95) -> str:
    """Return Q, I, or NQ under the frozen probability rule."""
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be finite and lie in [0, 1]")
    if probability >= confidence:
        return "Q"
    if probability <= 1.0 - confidence:
        return "NQ"
    return "I"


def add_p95_decision(
    frame: pd.DataFrame,
    probability_column: str = "estimated_probability",
    output_column: str = "decision",
) -> pd.DataFrame:
    """Add the frozen P95 state to an aggregate or safe-fixture table."""
    out = frame.copy()
    out[output_column] = out[probability_column].map(p95_state)
    return out


def aggregate_realization_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate safe realization-level Q/I/NQ records by setting and tolerance."""
    required = {"generator", "learner", "hospital_population", "delta", "policy", "decision"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    keys = ["generator", "learner", "hospital_population", "delta", "policy"]
    work = frame.copy()
    work["decision"] = work["decision"].replace(
        {"QUALIFIED": "Q", "INDETERMINATE": "I", "NOT_QUALIFIED": "NQ"}
    )
    if not set(work["decision"]).issubset({"Q", "I", "NQ"}):
        raise ValueError("decision must be Q, I, or NQ")
    table = pd.crosstab([work[k] for k in keys], work["decision"]).reset_index()
    for state in ("Q", "I", "NQ"):
        if state not in table:
            table[state] = 0
        table[f"{state}_n"] = table.pop(state).astype(int)
    table["denominator"] = table[["Q_n", "I_n", "NQ_n"]].sum(axis=1)
    for state in ("Q", "I", "NQ"):
        table[f"{state}_proportion"] = table[f"{state}_n"] / table["denominator"]
    return table[
        keys + ["denominator", "Q_n", "I_n", "NQ_n", "Q_proportion", "I_proportion", "NQ_proportion"]
    ].sort_values(keys).reset_index(drop=True)
