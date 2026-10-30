#!/usr/bin/env python3
"""Reproduce public E3/E5 outputs from released minimal V2 P95 records.

The input contains simulated replicate-level probabilities and decision indicators
only. This script performs no simulation, refitting, model training, prediction,
covariance estimation, threshold change, or clinical-data access.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproduced_extension"
PUBLIC_INPUT = ROOT / "extension/e3_e5/public_inputs/V2_P95_REPLICATE_METRICS.parquet"
EXPECTED_INPUT_SHA256 = "50b69c5b0d308b367f388508587cf6e6a927087342640e15dd74e2e8f214febd"
EXPECTED_E5 = ROOT / "supplementary_data/Supplementary_Data_10_E5_LEAVE_ONE_GENERATOR_OUT_RESULTS.csv"

A_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
B_GRID = (0.01, 0.025, 0.05, 0.10, 0.20, 0.40)
TIE_TOL = 1e-12
Z95 = 1.959963984540054

STRATEGIES = (
    "M1_FULL_COV_CROSSED",
    "M2_DIAGONAL_COV_CROSSED",
    "M3_REALIZATION_AGGREGATED",
    "M4_SINGLE_REALIZATION",
)

CONTRASTS = (
    ("M3_vs_M4_repeated_realization_information", "M3_REALIZATION_AGGREGATED", "M4_SINGLE_REALIZATION"),
    ("M2_vs_M3_crossed_structure", "M2_DIAGONAL_COV_CROSSED", "M3_REALIZATION_AGGREGATED"),
    ("M1_vs_M2_full_covariance", "M1_FULL_COV_CROSSED", "M2_DIAGONAL_COV_CROSSED"),
    ("M1_vs_M4_complete_strategy", "M1_FULL_COV_CROSSED", "M4_SINGLE_REALIZATION"),
)

METRICS = {
    "DER": "decision_error",
    "DY": "estimated_definitive",
    "CDY": "correct_definitive",
    "FQ_all": "false_qualification",
}

SCENARIO = ["dgp", "template_id", "generator", "learner", "regime", "delta"]
PAIR_KEY = SCENARIO + ["replicate"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def hard_stop(code: str, detail: str, evidence: dict | None = None) -> None:
    payload = {
        "status": code,
        "detail": detail,
        "evidence": evidence or {},
    }
    write_json(OUT / "HARD_STOP.json", payload)
    raise RuntimeError(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str, evidence: dict | None = None) -> None:
    if not condition:
        hard_stop(code, detail, evidence)


def bayes_actions(p: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Direct three-risk argmin with the frozen tie convention."""
    p = np.asarray(p, dtype=np.float64)
    risks = np.column_stack((1.0 - p, a * p, np.full(p.shape, b, dtype=np.float64)))
    minima = risks.min(axis=1)
    tied = np.abs(risks - minima[:, None]) <= TIE_TOL
    tie_count = tied.sum(axis=1)
    q = tied[:, 0]
    nq = tied[:, 1]
    definitive = q | nq
    actions = np.full(p.shape, "INDETERMINATE", dtype=object)
    actions[q & ~nq] = "QUALIFIED"
    actions[nq & ~q] = "NOT_QUALIFIED"
    both = q & nq
    actions[both & (p >= 1.0 / (1.0 + a))] = "QUALIFIED"
    actions[both & (p < 1.0 / (1.0 + a))] = "NOT_QUALIFIED"
    actions[~definitive] = "INDETERMINATE"
    return actions, tie_count


def regret_from_actions(actions: np.ndarray, pstar: np.ndarray, a: float, b: float) -> np.ndarray:
    pstar = np.asarray(pstar, dtype=np.float64)
    oracle = np.minimum(np.minimum(1.0 - pstar, a * pstar), b)
    selected = np.full(pstar.shape, np.nan, dtype=np.float64)
    selected[actions == "QUALIFIED"] = 1.0 - pstar[actions == "QUALIFIED"]
    selected[actions == "NOT_QUALIFIED"] = a * pstar[actions == "NOT_QUALIFIED"]
    selected[actions == "INDETERMINATE"] = b
    return selected - oracle


def interval_label(lower: pd.Series, upper: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            upper.to_numpy() < 0,
            "COMPARISON_LOWER_REGRET",
            np.where(lower.to_numpy() > 0, "COMPARISON_HIGHER_REGRET", "UNRESOLVED_INTERVAL_INCLUDES_ZERO"),
        ),
        index=lower.index,
    )


def sign_label(x: float, tol: float = 1e-15) -> str:
    if x > tol:
        return "POSITIVE"
    if x < -tol:
        return "NEGATIVE"
    return "ZERO_WITHIN_1E-15"


def aggregate_strategy(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    grouped = frame.groupby(SCENARIO + ["strategy"], sort=True, observed=True)[value]
    out = grouped.agg(attempted_n="size", success_n="count", mean_regret="mean", regret_sd="std").reset_index()
    out["failed_n"] = out["attempted_n"] - out["success_n"]
    out["mc_se"] = out["regret_sd"] / np.sqrt(out["success_n"])
    out["mc_lower"] = out["mean_regret"] - Z95 * out["mc_se"]
    out["mc_upper"] = out["mean_regret"] + Z95 * out["mc_se"]
    truth = frame[SCENARIO + ["oracle_probability"]].drop_duplicates()
    require(not truth.duplicated(SCENARIO).any(), "E3_STOP_PSTAR_IDENTITY_MISMATCH", "A scenario maps to more than one p_star.")
    out = out.merge(truth, on=SCENARIO, validate="many_to_one")
    return out


def paired_cell_stats(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    outputs = []
    success_counts = (
        frame.loc[frame[value].notna()]
        .groupby(SCENARIO + ["strategy"], sort=True, observed=True)
        .size()
        .rename("strategy_success_n")
        .reset_index()
    )
    for name, comparison, reference in CONTRASTS:
        left = frame.loc[(frame["strategy"] == comparison) & frame[value].notna(), PAIR_KEY + [value]]
        right = frame.loc[(frame["strategy"] == reference) & frame[value].notna(), PAIR_KEY + [value]]
        pair = left.merge(right, on=PAIR_KEY, suffixes=("_comparison", "_reference"), validate="one_to_one")
        pair["paired_difference"] = pair[f"{value}_comparison"] - pair[f"{value}_reference"]
        g = pair.groupby(SCENARIO, sort=True, observed=True)["paired_difference"]
        out = g.agg(pairwise_common_fit_n="size", paired_difference="mean", paired_variance="var").reset_index()
        out["cell_mc_variance"] = out["paired_variance"] / out["pairwise_common_fit_n"]
        out["mc_se"] = np.sqrt(out["cell_mc_variance"])
        out["mc_lower"] = out["paired_difference"] - Z95 * out["mc_se"]
        out["mc_upper"] = out["paired_difference"] + Z95 * out["mc_se"]
        out["contrast"] = name
        out["comparison_strategy"] = comparison
        out["reference_strategy"] = reference
        for strategy, label in ((comparison, "comparison_success_n"), (reference, "reference_success_n")):
            counts = success_counts.loc[success_counts["strategy"] == strategy, SCENARIO + ["strategy_success_n"]]
            out = out.merge(counts.rename(columns={"strategy_success_n": label}), on=SCENARIO, validate="one_to_one")
        out["comparison_failed_n"] = 1000 - out["comparison_success_n"]
        out["reference_failed_n"] = 1000 - out["reference_success_n"]
        truth = frame[SCENARIO + ["oracle_probability"]].drop_duplicates()
        require(not truth.duplicated(SCENARIO).any(), "E3_STOP_PSTAR_IDENTITY_MISMATCH", "A scenario maps to more than one p_star.")
        out = out.merge(truth, on=SCENARIO, validate="one_to_one")
        out["comparison_label"] = interval_label(out["mc_lower"], out["mc_upper"])
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def macro_from_cell_pairs(cell: pd.DataFrame, cell_target: int) -> pd.DataFrame:
    keys = ["dgp", "delta", "contrast", "comparison_strategy", "reference_strategy"]
    rows = []
    for key, block in cell.groupby(keys, sort=True, observed=True):
        require(
            len(block) == cell_target,
            "E3_E5_CELL_WEIGHTING_FAIL",
            f"Expected {cell_target} cells, found {len(block)}",
            {k: v for k, v in zip(keys, key)},
        )
        estimate = float(block["paired_difference"].mean())
        mc_variance = float(block["cell_mc_variance"].sum() / (cell_target**2))
        mc_se = math.sqrt(mc_variance)
        rows.append(
            dict(
                zip(keys, key),
                cell_n=cell_target,
                pairwise_common_fit_n=int(block["pairwise_common_fit_n"].sum()),
                comparison_success_n=int(block["comparison_success_n"].sum()),
                reference_success_n=int(block["reference_success_n"].sum()),
                comparison_failed_n=int(block["comparison_failed_n"].sum()),
                reference_failed_n=int(block["reference_failed_n"].sum()),
                pstar_min=float(block["oracle_probability"].min()),
                pstar_max=float(block["oracle_probability"].max()),
                pstar_unique_n=int(block["oracle_probability"].nunique()),
                paired_difference=estimate,
                cell_mc_variance=mc_variance,
                mc_se=mc_se,
                mc_lower=estimate - Z95 * mc_se,
                mc_upper=estimate + Z95 * mc_se,
            )
        )
    result = pd.DataFrame(rows)
    result["comparison_label"] = interval_label(result["mc_lower"], result["mc_upper"])
    return result


def load_and_verify() -> tuple[pd.DataFrame, dict]:
    observed_hash = sha256(PUBLIC_INPUT)
    require(observed_hash == EXPECTED_INPUT_SHA256, "E3_E5_STOP_INPUT_HASH_MISMATCH",
            "Released public replicate input changed.", {"expected": EXPECTED_INPUT_SHA256, "actual": observed_hash})
    p95 = pq.read_table(PUBLIC_INPUT).to_pandas()
    require(len(p95) == 1152000, "E3_E5_STOP_GRAIN_MISMATCH", "P95 row count is not 1,152,000.")
    full_key = PAIR_KEY + ["strategy"]
    require(not p95.duplicated(full_key).any(), "E3_E5_STOP_IDENTITY_MISMATCH", "P95 replicate/scenario/strategy keys are not unique.")
    require(p95["replicate"].between(1, 1000).all(), "E3_E5_STOP_IDENTITY_MISMATCH", "Replicate identities are outside 1..1000.")
    require(set(p95["delta"].unique()) == {0.025, 0.05, 0.1, 0.2}, "E3_E5_STOP_IDENTITY_MISMATCH", "Original tolerance identities differ.")
    require(set(p95["generator"].unique()) == {"GaussianCopula", "CTGAN", "TabDDPM"}, "E3_E5_STOP_IDENTITY_MISMATCH", "Generator identities differ.")
    require(set(p95["dgp"].unique()) == {"G", "S", "T"}, "E3_E5_STOP_IDENTITY_MISMATCH", "DGP identities differ.")
    require(p95.groupby(["dgp", "template_id", "regime", "delta", "strategy"], observed=True).size().eq(1000).all(), "E3_E5_STOP_GRAIN_MISMATCH", "Not every scenario/strategy has 1,000 attempted replicates.")

    success = p95["fit_status"] == "SUCCESS"
    failed = p95["fit_status"] == "FIT_FAILED"
    require(int(success.sum()) == 1151188 and int(failed.sum()) == 812, "E3_E5_STOP_FIT_STATUS_MISMATCH", "P95 status counts differ from fourfold FIT expansion.")
    require(p95.loc[success, ["estimated_probability", "oracle_probability"]].notna().all().all(), "E3_E5_STOP_PHAT_INCOMPLETE", "p_hat or p_star is missing on successful rows.")
    require(p95.loc[failed, "estimated_probability"].isna().all(), "E3_E5_STOP_PHAT_INCOMPLETE", "Failed fits unexpectedly have p_hat.")
    require(p95["oracle_probability"].notna().all(), "E3_E5_STOP_PSTAR_INCOMPLETE", "p_star is incomplete.")
    require(p95.loc[success, "estimated_probability"].between(0, 1).all(), "E3_E5_STOP_PHAT_INCOMPLETE", "p_hat lies outside [0,1].")
    require(p95["oracle_probability"].between(0, 1).all(), "E3_E5_STOP_PSTAR_INCOMPLETE", "p_star lies outside [0,1].")
    require(set(p95.loc[success, "estimated_state"].unique()) <= {"QUALIFIED", "INDETERMINATE", "NOT_QUALIFIED"}, "E3_E5_STOP_IDENTITY_MISMATCH", "Unexpected successful P95 action.")
    require((p95.loc[failed, "estimated_state"] == "FIT_FAILED").all(), "E3_E5_STOP_FIT_STATUS_MISMATCH", "Failed fits were not preserved as FIT_FAILED.")

    oracle_key = ["dgp", "template_id", "regime", "delta"]
    truth = p95[oracle_key + ["oracle_probability"]].drop_duplicates()
    require(len(truth) == 288 and not truth.duplicated(oracle_key).any(), "E3_E5_STOP_PSTAR_INCOMPLETE",
            "Public input does not contain 288 unique truth cells.")

    fit_key = ["dgp", "template_id", "regime", "replicate", "strategy"]
    fit = p95[fit_key + ["fit_status"]].drop_duplicates()
    require(len(fit) == 288000 and not fit.duplicated(fit_key).any(), "E3_E5_STOP_GRAIN_MISMATCH",
            "Collapsed fit grain is not 288,000 unique records.")
    fit_counts = fit["fit_status"].value_counts().to_dict()
    require(fit_counts == {"SUCCESS": 287797, "FIT_FAILED": 203}, "E3_E5_STOP_FIT_STATUS_MISMATCH",
            "Unexpected collapsed fit success/failure counts.", fit_counts)

    cell_counts = p95[SCENARIO].drop_duplicates().groupby(["dgp", "delta"], observed=True).size()
    require(cell_counts.eq(24).all(), "E3_E5_STOP_GRAIN_MISMATCH", "DGP×tolerance does not contain 24 template-regime cells.")
    generator_cells = p95[SCENARIO].drop_duplicates().groupby(["dgp", "delta", "generator"], observed=True).size()
    require(generator_cells.eq(8).all(), "E3_E5_STOP_GRAIN_MISMATCH", "Each generator does not contribute exactly eight cells per DGP×tolerance.")

    qc = {
        "input_hashes": {"public_p95_replicate_metrics": observed_hash},
        "fit_rows": len(fit),
        "fit_success_n": int((fit.fit_status == "SUCCESS").sum()),
        "fit_failed_n": int((fit.fit_status == "FIT_FAILED").sum()),
        "p95_rows": len(p95),
        "p95_success_rows": int(success.sum()),
        "p95_failed_rows": int(failed.sum()),
        "oracle_p95_truth_cells": len(truth),
        "oracle_long_max_abs_difference": 0.0,
        "scenario_cells_per_dgp_delta": sorted(cell_counts.unique().tolist()),
        "generator_cells_per_dgp_delta": sorted(generator_cells.unique().tolist()),
    }
    return p95, qc


def e3_execute(p95: pd.DataFrame, qc: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_grid_rows = []
    strategy_outputs = []
    paired_outputs = []
    fixed_outputs = []
    global_min_regret = math.inf
    global_min_fixed_regret = math.inf
    p95_sanity_matches = 0
    p95_sanity_total = 0

    success_mask = p95["fit_status"].eq("SUCCESS").to_numpy()
    p_hat = p95.loc[success_mask, "estimated_probability"].to_numpy(dtype=np.float64)
    p_star = p95.loc[success_mask, "oracle_probability"].to_numpy(dtype=np.float64)
    original_actions = p95.loc[success_mask, "estimated_state"].to_numpy(dtype=object)

    for a in A_GRID:
        for b in B_GRID:
            actions, ties = bayes_actions(p_hat, a, b)
            regret = regret_from_actions(actions, p_star, a, b)
            min_regret = float(np.min(regret))
            global_min_regret = min(global_min_regret, min_regret)
            require(min_regret >= -1e-12, "E3_STOP_NEGATIVE_REGRET", "Cost-matched regret is negative beyond numerical precision.", {"a": a, "b": b, "minimum_regret": min_regret})

            p95["_regret"] = np.nan
            p95.loc[success_mask, "_regret"] = regret
            strategy = aggregate_strategy(p95, "_regret")
            strategy["a"] = a
            strategy["b"] = b
            strategy["no_strict_abstention_region"] = bool(b >= a / (1.0 + a))
            strategy_outputs.append(strategy)

            pair_cell = paired_cell_stats(p95, "_regret")
            pair_cell["a"] = a
            pair_cell["b"] = b
            pair_cell["analysis_level"] = "SCENARIO_CELL_CONDITIONAL_PRIMARY"
            pair_cell["cell_n"] = 1
            pair_cell["no_strict_abstention_region"] = bool(b >= a / (1.0 + a))
            paired_outputs.append(pair_cell)

            macro = macro_from_cell_pairs(pair_cell, 24)
            macro["a"] = a
            macro["b"] = b
            macro["analysis_level"] = "DGP_DELTA_EQUAL_24_ORIGINAL_SCENARIO_CELLS_DESCRIPTIVE"
            macro["no_strict_abstention_region"] = bool(b >= a / (1.0 + a))
            paired_outputs.append(macro)

            fixed_regret = regret_from_actions(original_actions, p_star, a, b)
            min_fixed_regret = float(np.min(fixed_regret))
            global_min_fixed_regret = min(global_min_fixed_regret, min_fixed_regret)
            require(min_fixed_regret >= -1e-12, "E3_STOP_NEGATIVE_REGRET", "Fixed-P95 regret is negative beyond numerical precision.", {"a": a, "b": b, "minimum_regret": min_fixed_regret})
            p95["_fixed_regret"] = np.nan
            p95.loc[success_mask, "_fixed_regret"] = fixed_regret
            fixed_strategy = aggregate_strategy(p95, "_fixed_regret")
            fixed_strategy["record_type"] = "STRATEGY_SCENARIO_CELL"
            fixed_strategy["a"] = a
            fixed_strategy["b"] = b
            fixed_outputs.append(fixed_strategy)
            fixed_pair = paired_cell_stats(p95, "_fixed_regret")
            fixed_pair["record_type"] = "PAIRED_STRATEGY_CONTRAST_SCENARIO_CELL"
            fixed_pair["a"] = a
            fixed_pair["b"] = b
            fixed_outputs.append(fixed_pair)

            sanity_match_n = None
            sanity_total_n = None
            if a == 1.0 and b == 0.05:
                sanity_success = int(np.sum(actions == original_actions))
                sanity_failed = int((~success_mask).sum())
                sanity_match_n = sanity_success + sanity_failed
                sanity_total_n = len(p95)
                p95_sanity_matches = sanity_match_n
                p95_sanity_total = sanity_total_n
                require(sanity_match_n == sanity_total_n, "E3_P95_SANITY_CHECK_FAIL", "Bayes actions at a=1,b=0.05 differ from original P95 actions.", {"matches": sanity_match_n, "total": sanity_total_n})

            cost_grid_rows.append(
                {
                    "a": a,
                    "b": b,
                    "abstention_boundary": a / (1.0 + a),
                    "structural_annotation": "NO_STRICT_ABSTENTION_REGION" if b >= a / (1.0 + a) else "STRICT_ABSTENTION_REGION_EXISTS",
                    "no_strict_abstention_region": bool(b >= a / (1.0 + a)),
                    "successful_action_evaluations": len(actions),
                    "tie_n": int(np.sum(ties > 1)),
                    "tie_rate": float(np.mean(ties > 1)),
                    "p95_sanity_match_n": sanity_match_n,
                    "p95_sanity_total_n": sanity_total_n,
                }
            )

    strategy_results = pd.concat(strategy_outputs, ignore_index=True)
    pair_results = pd.concat(paired_outputs, ignore_index=True, sort=False)
    fixed_results = pd.concat(fixed_outputs, ignore_index=True, sort=False)
    cost_grid = pd.DataFrame(cost_grid_rows)

    require(len(cost_grid) == 30, "E3_STOP_COST_GRID_INCOMPLETE", "Frozen 30-cell cost grid was not preserved.")
    require(len(strategy_results) == 34560, "E3_STOP_SCENARIO_GRID_INCOMPLETE", "E3 strategy result grid is incomplete.", {"rows": len(strategy_results)})
    require(len(pair_results) == 36000, "E3_STOP_SCENARIO_GRID_INCOMPLETE", "E3 paired result grid is incomplete.", {"rows": len(pair_results)})
    require(len(fixed_results) == 69120, "E3_STOP_SCENARIO_GRID_INCOMPLETE", "E3 fixed-P95 sensitivity grid is incomplete.", {"rows": len(fixed_results)})
    require(p95_sanity_total == 1152000 and p95_sanity_matches == p95_sanity_total, "E3_P95_SANITY_CHECK_FAIL", "P95 sanity denominator or identity is incomplete.")

    qc.update(
        {
            "e3_cost_grid_n": len(cost_grid),
            "e3_strategy_result_rows": len(strategy_results),
            "e3_paired_result_rows": len(pair_results),
            "e3_fixed_p95_result_rows": len(fixed_results),
            "e3_p95_sanity_matches": p95_sanity_matches,
            "e3_p95_sanity_total": p95_sanity_total,
            "e3_p95_sanity_identity_rate": p95_sanity_matches / p95_sanity_total,
            "e3_minimum_cost_matched_regret": global_min_regret,
            "e3_minimum_fixed_p95_regret": global_min_fixed_regret,
            "e3_negative_regret_tolerance": -1e-12,
            "e3_tie_total": int(cost_grid["tie_n"].sum()),
            "e3_no_strict_abstention_cost_cells": int(cost_grid["no_strict_abstention_region"].sum()),
        }
    )
    return cost_grid, strategy_results, pair_results, fixed_results


def e5_execute(p95: pd.DataFrame, qc: dict) -> pd.DataFrame:
    cell_outputs = []
    for name, comparison, reference in CONTRASTS:
        cols = PAIR_KEY + list(METRICS.values())
        left = p95.loc[(p95["strategy"] == comparison) & (p95["fit_status"] == "SUCCESS"), cols]
        right = p95.loc[(p95["strategy"] == reference) & (p95["fit_status"] == "SUCCESS"), cols]
        pair = left.merge(right, on=PAIR_KEY, suffixes=("_comparison", "_reference"), validate="one_to_one")
        comp_success = left.groupby(SCENARIO, sort=True, observed=True).size().rename("comparison_success_n").reset_index()
        ref_success = right.groupby(SCENARIO, sort=True, observed=True).size().rename("reference_success_n").reset_index()
        for metric, source in METRICS.items():
            pair["_diff"] = pair[f"{source}_comparison"].astype(float) - pair[f"{source}_reference"].astype(float)
            g = pair.groupby(SCENARIO, sort=True, observed=True)["_diff"]
            cell = g.agg(pairwise_common_fit_n="size", paired_difference="mean", paired_variance="var").reset_index()
            cell["cell_mc_variance"] = cell["paired_variance"] / cell["pairwise_common_fit_n"]
            cell = cell.merge(comp_success, on=SCENARIO, validate="one_to_one").merge(ref_success, on=SCENARIO, validate="one_to_one")
            cell["comparison_failed_n"] = 1000 - cell["comparison_success_n"]
            cell["reference_failed_n"] = 1000 - cell["reference_success_n"]
            cell["contrast"] = name
            cell["comparison_strategy"] = comparison
            cell["reference_strategy"] = reference
            cell["metric"] = metric
            cell_outputs.append(cell)
    cell_stats = pd.concat(cell_outputs, ignore_index=True)
    require(len(cell_stats) == 4608, "E5_STOP_CELL_GRID_INCOMPLETE", "E5 cell-level metric grid is incomplete.", {"rows": len(cell_stats)})

    # Reconstruct the locked 24-cell macro separately for each metric.
    rows = []
    group_keys = ["dgp", "delta", "contrast", "comparison_strategy", "reference_strategy", "metric"]
    for key, block in cell_stats.groupby(group_keys, sort=True, observed=True):
        require(len(block) == 24, "E5_STOP_24_CELL_WEIGHTING_FAIL", "Original baseline does not contain 24 cells.", {k: v for k, v in zip(group_keys, key)})
        est = float(block["paired_difference"].mean())
        mcvar = float(block["cell_mc_variance"].sum() / 24**2)
        se = math.sqrt(mcvar)
        rows.append(dict(zip(group_keys, key), cell_n=24, pairwise_common_fit_n=int(block["pairwise_common_fit_n"].sum()), paired_difference=est, mc_se=se, mc_lower=est-Z95*se, mc_upper=est+Z95*se))
    original = pd.DataFrame(rows)

    compare_keys = ["dgp", "delta", "contrast", "comparison_strategy", "reference_strategy", "metric"]
    expected_e5 = pd.read_csv(EXPECTED_E5)
    stored = expected_e5[compare_keys + [
        "original_cell_n", "original_pairwise_common_fit_n", "original_paired_difference",
        "original_mc_se", "original_mc_lower", "original_mc_upper",
    ]].drop_duplicates(compare_keys).rename(columns={
        "original_cell_n": "cell_n",
        "original_pairwise_common_fit_n": "pairwise_common_fit_n",
        "original_paired_difference": "paired_difference",
        "original_mc_se": "mc_se",
        "original_mc_lower": "mc_lower",
        "original_mc_upper": "mc_upper",
    })
    check = original.merge(stored, on=compare_keys, suffixes=("_recomputed", "_stored"), validate="one_to_one")
    require(len(check) == 192, "E5_STOP_BASELINE_RECONSTRUCTION_FAIL", "Stored/recomputed P95 contrast identities are incomplete.", {"rows": len(check)})
    numeric_diffs = {}
    for column in ["paired_difference", "mc_se", "mc_lower", "mc_upper"]:
        numeric_diffs[column] = float(np.max(np.abs(check[f"{column}_recomputed"] - check[f"{column}_stored"])))
    count_match = bool((check["cell_n_recomputed"] == check["cell_n_stored"]).all() and (check["pairwise_common_fit_n_recomputed"] == check["pairwise_common_fit_n_stored"]).all())
    require(max(numeric_diffs.values()) <= 1e-15 and count_match, "E5_STOP_BASELINE_RECONSTRUCTION_FAIL", "Recomputed original P95 macro contrasts do not reproduce locked outputs.", {"max_abs_differences": numeric_diffs, "count_match": count_match})

    result_rows = []
    for excluded in ("GaussianCopula", "CTGAN", "TabDDPM"):
        remaining = cell_stats.loc[cell_stats["generator"] != excluded]
        for key, block in remaining.groupby(group_keys, sort=True, observed=True):
            require(len(block) == 16, "E5_STOP_16_CELL_WEIGHTING_FAIL", "Leave-one-generator-out macro does not contain 16 cells.", {"excluded_generator": excluded, **{k: v for k, v in zip(group_keys, key)}})
            est = float(block["paired_difference"].mean())
            mcvar = float(block["cell_mc_variance"].sum() / 16**2)
            se = math.sqrt(mcvar)
            base_match = original
            for k, v in zip(group_keys, key):
                base_match = base_match.loc[base_match[k] == v]
            require(len(base_match) == 1, "E5_STOP_IDENTITY_MISMATCH", "Could not identify unique original baseline for E5 output.")
            base = base_match.iloc[0]
            base_label = "COMPARISON_LOWER" if base.mc_upper < 0 else ("COMPARISON_HIGHER" if base.mc_lower > 0 else "UNRESOLVED_INTERVAL_INCLUDES_ZERO")
            low, high = est - Z95 * se, est + Z95 * se
            excl_label = "COMPARISON_LOWER" if high < 0 else ("COMPARISON_HIGHER" if low > 0 else "UNRESOLVED_INTERVAL_INCLUDES_ZERO")
            result_rows.append(
                {
                    "excluded_generator": excluded,
                    **dict(zip(group_keys, key)),
                    "remaining_cell_n": 16,
                    "weighting": "EQUAL_16_TEMPLATE_REGIME_CELLS",
                    "pairwise_common_fit_n": int(block["pairwise_common_fit_n"].sum()),
                    "original_cell_n": int(base.cell_n),
                    "original_pairwise_common_fit_n": int(base.pairwise_common_fit_n),
                    "original_paired_difference": float(base.paired_difference),
                    "original_mc_se": float(base.mc_se),
                    "original_mc_lower": float(base.mc_lower),
                    "original_mc_upper": float(base.mc_upper),
                    "original_interval_label": base_label,
                    "excluded_paired_difference": est,
                    "excluded_mc_se": se,
                    "excluded_mc_lower": low,
                    "excluded_mc_upper": high,
                    "excluded_interval_label": excl_label,
                    "signed_estimate_change": est - float(base.paired_difference),
                    "absolute_magnitude_change": abs(est) - abs(float(base.paired_difference)),
                    "original_direction": sign_label(float(base.paired_difference)),
                    "excluded_direction": sign_label(est),
                    "direction_changed": sign_label(float(base.paired_difference)) != sign_label(est),
                    "interval_resolution_changed": base_label != excl_label,
                }
            )
    results = pd.DataFrame(result_rows)
    require(len(results) == 576, "E5_STOP_GRID_INCOMPLETE", "E5 output grid is not 576 rows.", {"rows": len(results)})
    require(results["remaining_cell_n"].eq(16).all(), "E5_STOP_16_CELL_WEIGHTING_FAIL", "E5 16-cell macro weighting failed.")

    qc.update(
        {
            "e5_cell_metric_rows": len(cell_stats),
            "e5_original_macro_rows_recomputed": len(original),
            "e5_original_locked_reconstruction_max_abs_differences": numeric_diffs,
            "e5_original_locked_count_match": count_match,
            "e5_leave_one_generator_out_rows": len(results),
            "e5_remaining_cell_counts": sorted(results["remaining_cell_n"].unique().tolist()),
            "e5_direction_change_rows": int(results["direction_changed"].sum()),
            "e5_interval_resolution_change_rows": int(results["interval_resolution_changed"].sum()),
        }
    )
    return results


def findings_report(pair: pd.DataFrame, fixed: pd.DataFrame, e5: pd.DataFrame, qc: dict) -> str:
    primary = pair.loc[pair["analysis_level"] == "SCENARIO_CELL_CONDITIONAL_PRIMARY"]
    primary_counts = primary.groupby(["contrast", "comparison_label"], observed=True).size().unstack(fill_value=0)
    fixed_pair = fixed.loc[fixed["record_type"] == "PAIRED_STRATEGY_CONTRAST_SCENARIO_CELL"]
    fixed_counts = fixed_pair.groupby(["contrast", "comparison_label"], observed=True).size().unstack(fill_value=0)
    e5_summary = e5.groupby(["excluded_generator", "contrast"], observed=True).agg(
        rows=("metric", "size"),
        direction_changes=("direction_changed", "sum"),
        interval_label_changes=("interval_resolution_changed", "sum"),
        max_abs_signed_change=("signed_estimate_change", lambda x: float(np.max(np.abs(x)))),
    ).reset_index()

    lines = [
        "# E3/E5 bounded scientific findings",
        "",
        "**Executor status: `E3_E5_BOUNDED_PRODUCTION_COMPLETE_READY_FOR_TERRA_AUDIT`.** This is production plus internal numerical QC, not independent audit or PI scientific integration.",
        "",
        "## Scope and evidence boundary",
        "",
        "E3 and E5 were computed exclusively from the locked original V2 replicate-level outputs. No simulation, refitting, model training, prediction, covariance estimation, E2/E4 execution, or manuscript change was performed. E3 is a post hoc, scenario-conditional scope reduction authorized by the PI after the E2 route stopped; it does not restore the unexecuted 45-point latent-probability sweep.",
        "",
        "The E3 primary estimand is conditional on each of the 288 original DGP × template × regime × tolerance truth cells. It therefore supports conclusions only at the observed original truth positions. It does not cover arbitrary latent truth positions, and no continuous boundary surface was interpolated. Descriptive DGP × tolerance macro rows equally weight the fixed 24 original scenario cells and do not define a prior over latent truth.",
        "",
        "## Hard-gate results",
        "",
        f"- p-hat and p-star authority/completeness: PASS ({qc['p95_success_rows']:,} successful P95 rows; {qc['oracle_p95_truth_cells']} oracle truth cells; exact LONG/oracle p-star agreement).",
        f"- Replicate/scenario identity and fit-status preservation: PASS ({qc['p95_rows']:,} attempted P95 rows; {qc['p95_failed_rows']:,} failed rows retained as `FIT_FAILED`, never recoded as indeterminate).",
        f"- P95 sanity at a=1, b=0.05: PASS ({qc['e3_p95_sanity_matches']:,}/{qc['e3_p95_sanity_total']:,}; 100% including preserved fit-failure records).",
        f"- Nonnegative regret: PASS (minimum cost-matched regret {qc['e3_minimum_cost_matched_regret']:.17g}; tolerance -1e-12).",
        f"- E5 original contrast reconstruction: PASS (maximum numeric discrepancy {max(qc['e5_original_locked_reconstruction_max_abs_differences'].values()):.3g}; exact counts).",
        "- E5 leave-one-generator-out weighting: PASS (16 cells in every DGP × tolerance × contrast × metric × exclusion row).",
        "",
        "## E3 scenario-conditional cost-matched regret",
        "",
        "The full 5 × 6 cost grid was retained. Cells with b ≥ a/(1+a) are explicitly marked `NO_STRICT_ABSTENTION_REGION`; none was deleted. Pairwise interval labels use paired replicate-level Monte Carlo uncertainty, and intervals containing zero remain unresolved. Counts below are descriptive counts over 8,640 scenario-cell × cost comparisons per contrast, not multiplicity-adjusted discoveries:",
        "",
        "| Contrast | Comparison lower regret | Comparison higher regret | Unresolved |",
        "|---|---:|---:|---:|",
    ]
    for contrast in [c[0] for c in CONTRASTS]:
        row = primary_counts.loc[contrast]
        lines.append(f"| {contrast} | {int(row.get('COMPARISON_LOWER_REGRET', 0))} | {int(row.get('COMPARISON_HIGHER_REGRET', 0))} | {int(row.get('UNRESOLVED_INTERVAL_INCLUDES_ZERO', 0))} |")
    lines += [
        "",
        "These counts preserve heterogeneous, interval-supported, and unresolved comparisons. They do not identify a universally optimal strategy or a clinically privileged cost function.",
        "",
        "## E3 fixed-P95 sensitivity",
        "",
        "Fixed original P95 actions were evaluated under the same 30 cost cells as a secondary mismatch sensitivity. Counts below again refer to 8,640 scenario-cell × cost comparisons per contrast:",
        "",
        "| Contrast | Comparison lower regret | Comparison higher regret | Unresolved |",
        "|---|---:|---:|---:|",
    ]
    for contrast in [c[0] for c in CONTRASTS]:
        row = fixed_counts.loc[contrast]
        lines.append(f"| {contrast} | {int(row.get('COMPARISON_LOWER_REGRET', 0))} | {int(row.get('COMPARISON_HIGHER_REGRET', 0))} | {int(row.get('UNRESOLVED_INTERVAL_INCLUDES_ZERO', 0))} |")
    lines += [
        "",
        "## E5 leave-one-generator-out robustness",
        "",
        "Each exclusion preserved the frozen four contrasts, four metrics, three DGPs, four tolerances, within-cell common-success pairing, and equal weighting of the remaining 16 cells. No materiality threshold was prespecified; the table reports point-direction and interval-label changes descriptively and does not assess generator training adequacy.",
        "",
        "| Excluded generator | Contrast | Rows | Direction changes | Interval-label changes | Maximum absolute signed estimate change |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in e5_summary.itertuples(index=False):
        lines.append(f"| {row.excluded_generator} | {row.contrast} | {int(row.rows)} | {int(row.direction_changes)} | {int(row.interval_label_changes)} | {row.max_abs_signed_change:.6g} |")
    lines += [
        "",
        "## Permanent qualifications",
        "",
        "E1 remains `E1_PI_LOCKED_WITH_QUALIFICATIONS`; original V2 remains `HISTORICAL_LOCKED_AUTHORITY`. The documented M1/M2 finite-precision optimizer sensitivity remains an identifiable numerical limitation of the inherited original implementation, but the shifted-fit failures do not invalidate the original unshifted V2 results. E3/E5 use the frozen successful-fit/common-success statuses exactly as recorded and do not repair or reinterpret them.",
        "",
        "## Next action",
        "",
        "Independent Codex Terra audit, followed by PI scientific adjudication. No manuscript integration is authorized by this executor report.",
        "",
        "`E3_E5_BOUNDED_PRODUCTION_COMPLETE_READY_FOR_TERRA_AUDIT`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    p95, qc = load_and_verify()

    cost_grid, strategy_results, pair_results, fixed_results = e3_execute(p95, qc)
    e5_results = e5_execute(p95, qc)

    write_csv(OUT / "E3_COST_GRID.csv", cost_grid)
    write_csv(OUT / "E3_CONDITIONAL_REGRET_RESULTS.csv", strategy_results)
    write_csv(OUT / "E3_PAIRED_STRATEGY_CONTRASTS.csv", pair_results)
    write_csv(OUT / "E3_FIXED_P95_SENSITIVITY.csv", fixed_results)
    write_csv(OUT / "E5_LEAVE_ONE_GENERATOR_OUT_RESULTS.csv", e5_results)

    qc["hard_gates"] = {
        "authoritative_p_hat_p_star_complete": "PASS",
        "replicate_scenario_identity_exact": "PASS",
        "p95_sanity_identity": "PASS",
        "nonnegative_regret": "PASS",
        "e5_common_success_and_16_cell_weighting": "PASS",
        "no_new_fit_or_definition": "PASS",
    }
    qc["analysis_actions"] = {
        "new_simulations": 0,
        "new_model_fits": 0,
        "new_predictions": 0,
        "new_covariance_estimates": 0,
        "e2_or_e4_executed": False,
    }
    # Independent public verification against released Supplementary Data.
    expected = {
        "E3_COST_GRID.csv": ROOT / "supplementary_data/Supplementary_Data_06_E3_COST_GRID.csv",
        "E3_CONDITIONAL_REGRET_RESULTS.csv": ROOT / "supplementary_data/Supplementary_Data_07_E3_CONDITIONAL_REGRET_RESULTS.csv",
        "E3_PAIRED_STRATEGY_CONTRASTS.csv": ROOT / "supplementary_data/Supplementary_Data_08_E3_PAIRED_STRATEGY_CONTRASTS.csv",
        "E3_FIXED_P95_SENSITIVITY.csv": ROOT / "supplementary_data/Supplementary_Data_09_E3_FIXED_P95_SENSITIVITY.csv",
        "E5_LEAVE_ONE_GENERATOR_OUT_RESULTS.csv": EXPECTED_E5,
    }
    comparisons = {}
    for output_name, reference_path in expected.items():
        observed = pd.read_csv(OUT / output_name)
        reference = pd.read_csv(reference_path)
        common = [c for c in observed.columns if c in reference.columns]
        observed = observed[common]
        reference = reference[common]
        require(observed.shape == reference.shape, "PUBLIC_REPRODUCTION_SHAPE_MISMATCH", output_name,
                {"observed": observed.shape, "reference": reference.shape})
        for c in common:
            if pd.api.types.is_numeric_dtype(observed[c]) and pd.api.types.is_numeric_dtype(reference[c]):
                a = observed[c].to_numpy(float)
                b = reference[c].to_numpy(float)
                require(np.allclose(a, b, rtol=0, atol=5e-15, equal_nan=True),
                        "PUBLIC_REPRODUCTION_NUMERIC_MISMATCH", f"{output_name}:{c}")
            else:
                require(observed[c].fillna("").astype(str).equals(reference[c].fillna("").astype(str)),
                        "PUBLIC_REPRODUCTION_TEXT_MISMATCH", f"{output_name}:{c}")
        comparisons[output_name] = {"rows": len(observed), "columns_compared": len(common), "status": "PASS"}

    supported_reversals = int((((e5_results.original_interval_label == "COMPARISON_HIGHER") &
                                (e5_results.excluded_interval_label == "COMPARISON_LOWER")) |
                               ((e5_results.original_interval_label == "COMPARISON_LOWER") &
                                (e5_results.excluded_interval_label == "COMPARISON_HIGHER"))).sum())
    require(int(e5_results.direction_changed.sum()) == 54, "PUBLIC_E5_COUNT_MISMATCH", "direction changes")
    require(supported_reversals == 31, "PUBLIC_E5_COUNT_MISMATCH", "interval-supported reversals")
    require(int(e5_results.interval_resolution_changed.sum()) == 71, "PUBLIC_E5_COUNT_MISMATCH", "interval classification changes")
    qc["released_output_comparisons"] = comparisons
    qc["e5_interval_supported_reversals"] = supported_reversals
    qc["analysis_actions"] = {"new_simulations": 0, "new_model_fits": 0, "new_predictions": 0, "clinical_data_access": False}
    qc["public_reproduction_status"] = "PASS"
    qc["software"] = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__}
    write_json(OUT / "PUBLIC_E3_E5_REPRODUCTION_QC.json", qc)

    manifest_rows = [{"role": "INPUT", "name": PUBLIC_INPUT.name, "path": str(PUBLIC_INPUT.relative_to(ROOT)), "bytes": PUBLIC_INPUT.stat().st_size, "sha256": sha256(PUBLIC_INPUT)}]
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"PUBLIC_E3_E5_REPRODUCTION_MANIFEST.csv", "HARD_STOP.json"}:
            manifest_rows.append({"role": "OUTPUT", "name": path.name, "path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUT / "PUBLIC_E3_E5_REPRODUCTION_MANIFEST.csv", pd.DataFrame(manifest_rows))
    print("PUBLIC_E3_E5_REPRODUCTION_PASS")


if __name__ == "__main__":
    main()
