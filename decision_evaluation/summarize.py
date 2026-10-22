#!/usr/bin/env python3
"""Frozen decision, contrast, coverage, and fit-accounting summaries."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "reproduction_results"
SIM = PROJECT / "simulation"
QC = PROJECT / "qc"
FIT_PATH = SIM / "STUDY_FIT_RESULTS.parquet"
LONG_PATH = SIM / "STUDY_SIMULATION_LONG.parquet"
METHODS = ["M1_FULL_COV_CROSSED", "M2_DIAGONAL_COV_CROSSED", "M3_REALIZATION_AGGREGATED", "M4_SINGLE_REALIZATION"]
CONTRASTS = [
    ("M3_vs_M4_repeated_realization_information", METHODS[2], METHODS[3]),
    ("M2_vs_M3_crossed_structure", METHODS[1], METHODS[2]),
    ("M1_vs_M2_full_covariance", METHODS[0], METHODS[1]),
    ("M1_vs_M4_complete_strategy", METHODS[0], METHODS[3]),
]
BINARY = {
    "DER": "decision_error",
    "DY": "estimated_definitive",
    "FQ_all": "false_qualification",
    "FR_all": "false_rejection",
    "CDY": "correct_definitive",
    "under_recognition": "under_recognition_uncertainty",
    "excess_indeterminacy": "excess_indeterminacy",
}


def cell_metric_rows(d: pd.DataFrame, keys: list[str]) -> list[dict]:
    rows: list[dict] = []
    for key, q in d.groupby(keys, sort=True, observed=True):
        q = q[q.fit_status == "SUCCESS"]
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        n = len(q)
        if n == 0:
            continue
        for metric, column in BINARY.items():
            x = q[column].astype(float)
            p = float(x.mean())
            rows.append(base | {"metric": metric, "value": p, "mc_variance": p * (1 - p) / n, "evaluable_n": n, "denominator_n": n})
        definitive = q.estimated_definitive.astype(bool)
        dn = int(definitive.sum())
        for metric, column in (("SER", "selective_error"), ("FQ_def", "false_qualification")):
            if dn:
                p = float(q.loc[definitive, column].astype(float).mean())
                var = p * (1 - p) / dn
            else:
                p, var = math.nan, math.nan
            rows.append(base | {"metric": metric, "value": p, "mc_variance": var, "evaluable_n": n, "denominator_n": dn})
        for oracle in ("QUALIFIED", "INDETERMINATE", "NOT_QUALIFIED"):
            for estimated in ("QUALIFIED", "INDETERMINATE", "NOT_QUALIFIED"):
                x = ((q.oracle_state == oracle) & (q.estimated_state == estimated)).astype(float)
                p = float(x.mean())
                rows.append(base | {"metric": f"CONFUSION_{oracle}_TO_{estimated}", "value": p, "mc_variance": p * (1 - p) / n, "evaluable_n": n, "denominator_n": n})
    return rows


def macro_from_cells(cells: pd.DataFrame, macro_keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, q in cells.groupby(macro_keys + ["metric"], sort=True, observed=True):
        base = dict(zip(macro_keys + ["metric"], key))
        z = q.value.dropna()
        k = len(z)
        if k == 0:
            value = se = math.nan
        else:
            value = float(z.mean())
            aligned = q.loc[z.index, "mc_variance"].fillna(0).astype(float)
            se = float(math.sqrt(aligned.sum()) / k)
        rows.append(base | {
            "cell_n": k,
            "macro_value": value,
            "mc_se": se,
            "mc_lower": max(0.0, value - 1.959963984540054 * se) if math.isfinite(value) else math.nan,
            "mc_upper": min(1.0, value + 1.959963984540054 * se) if math.isfinite(value) else math.nan,
            "successful_fit_n": int(q.evaluable_n.sum()),
            "metric_denominator_n": int(q.denominator_n.sum()),
            "macro_weighting": "EQUAL_WEIGHT_ACROSS_24_TEMPLATE_REGIME_CELLS",
        })
    return pd.DataFrame(rows)


def paired_metric(q: pd.DataFrame, a: str, b: str, metric: str) -> tuple[float, float, int]:
    qa = q[q.strategy == a].set_index("replicate")
    qb = q[q.strategy == b].set_index("replicate")
    ids = qa.index.intersection(qb.index)
    qa, qb = qa.loc[ids], qb.loc[ids]
    n = len(ids)
    if n == 0:
        return math.nan, math.nan, 0
    if metric in BINARY:
        xa = qa[BINARY[metric]].astype(float).to_numpy()
        xb = qb[BINARY[metric]].astype(float).to_numpy()
        diff = xa - xb
        return float(diff.mean()), float(diff.var(ddof=1) / n) if n > 1 else 0.0, n
    if metric in ("SER", "FQ_def"):
        numerator = "selective_error" if metric == "SER" else "false_qualification"
        da = qa.estimated_definitive.astype(float).to_numpy()
        db = qb.estimated_definitive.astype(float).to_numpy()
        na = qa[numerator].astype(float).to_numpy()
        nb = qb[numerator].astype(float).to_numpy()
        ma, mb = da.mean(), db.mean()
        if ma == 0 or mb == 0:
            return math.nan, math.nan, n
        ra, rb = na.mean() / ma, nb.mean() / mb
        influence = (na - ra * da) / ma - (nb - rb * db) / mb
        return float(ra - rb), float(influence.var(ddof=1) / n) if n > 1 else 0.0, n
    raise ValueError(metric)


def make_contrasts(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["dgp", "template_id", "regime", "delta", "policy", "confidence"]
    cell_rows = []
    successful = long[long.fit_status == "SUCCESS"]
    for key, d in successful.groupby(keys, sort=True, observed=True):
        base = dict(zip(keys, key))
        for name, a, b in CONTRASTS:
            for metric in list(BINARY) + ["SER", "FQ_def"]:
                value, variance, n = paired_metric(d, a, b, metric)
                cell_rows.append(base | {"contrast": name, "reference_strategy": b, "comparison_strategy": a, "metric": metric, "cell_difference": value, "mc_variance": variance, "pairwise_common_n": n})
    cell = pd.DataFrame(cell_rows)
    rows = []
    macro_keys = ["dgp", "delta", "policy", "confidence", "contrast", "reference_strategy", "comparison_strategy", "metric"]
    for key, q in cell.groupby(macro_keys, sort=True, observed=True):
        base = dict(zip(macro_keys, key))
        z = q.cell_difference.dropna()
        k = len(z)
        value = float(z.mean()) if k else math.nan
        se = float(math.sqrt(q.loc[z.index, "mc_variance"].fillna(0).sum()) / k) if k else math.nan
        rows.append(base | {"cell_n": k, "paired_difference": value, "mc_se": se, "mc_lower": value - 1.959963984540054 * se if k else math.nan, "mc_upper": value + 1.959963984540054 * se if k else math.nan, "pairwise_common_fit_n": int(q.pairwise_common_n.sum())})
    return pd.DataFrame(rows)


def four_way_summary(long: pd.DataFrame) -> pd.DataFrame:
    status = long[["dgp", "template_id", "regime", "replicate", "strategy", "fit_status"]].drop_duplicates()
    common = status.assign(ok=status.fit_status.eq("SUCCESS")).groupby(["dgp", "template_id", "regime", "replicate"], as_index=False).ok.sum()
    common = common[common.ok == 4].drop(columns="ok")
    d = long.merge(common, on=["dgp", "template_id", "regime", "replicate"], how="inner", validate="many_to_one")
    keys = ["dgp", "template_id", "regime", "strategy", "delta", "policy", "confidence"]
    cell = pd.DataFrame(cell_metric_rows(d, keys))
    macro = macro_from_cells(cell, ["dgp", "strategy", "delta", "policy", "confidence"])
    macro["sensitivity_population"] = "FOUR_WAY_COMMON_SUCCESS"
    return macro


def coverage_outputs(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, recovery = [], []
    keys = ["dgp", "template_id", "generator", "learner", "regime", "strategy"]
    for key, d in fits.groupby(keys, sort=True, observed=True):
        ok = d[d.fit_status == "SUCCESS"]
        base = dict(zip(keys, key)) | {"attempted_n": len(d), "successful_n": len(ok), "failed_n": len(d) - len(ok), "failure_rate": (len(d) - len(ok)) / len(d)}
        for target in ("A", "B"):
            if len(ok):
                covered = ok[f"target_{target}_covered"].astype(bool)
                rows.append(base | {"target": target, "coverage": float(covered.mean()), "mean_interval_width": float(ok[f"target_{target}_pi_width"].mean()), "nominal": 0.95})
        if len(ok):
            for component in ("seed", "hospital", "interaction"):
                col = f"sigma2_{component}_estimate"
                truth_col = f"sigma2_{component}_truth"
                x = ok[col].dropna().astype(float)
                if len(x):
                    truth = float(ok[truth_col].iloc[0])
                    recovery.append(base | {"component": component, "truth": truth, "defined_n": len(x), "mean_estimate": float(x.mean()), "median_estimate": float(x.median()), "bias": float(x.mean() - truth), "rmse": float(np.sqrt(np.mean((x - truth) ** 2)))})
    return pd.DataFrame(rows), pd.DataFrame(recovery)


def main() -> None:
    fits = pd.read_parquet(FIT_PATH)
    long = pd.read_parquet(LONG_PATH, columns=[
        "dgp", "template_id", "generator", "learner", "regime", "replicate", "strategy",
        "delta", "policy", "confidence", "oracle_state", "estimated_state", "fit_status",
        "decision_error", "estimated_definitive", "selective_error", "false_qualification",
        "false_rejection", "correct_definitive", "under_recognition_uncertainty", "excess_indeterminacy",
    ])
    expected_fits, expected_long = 72_000 * 4, 72_000 * 4 * 4 * 7
    if len(fits) != expected_fits or len(long) != expected_long:
        raise RuntimeError(f"denominator mismatch fits={len(fits)} long={len(long)}")

    fit_account = fits.groupby(["dgp", "template_id", "generator", "learner", "regime", "strategy"], as_index=False).agg(
        attempted_n=("fit_status", "size"), successful_n=("fit_status", lambda x: int((x == "SUCCESS").sum())), failed_n=("fit_status", lambda x: int((x != "SUCCESS").sum()))
    )
    fit_account["failure_rate"] = fit_account.failed_n / fit_account.attempted_n
    fit_account["estimation_instability"] = fit_account.failure_rate > .05
    fit_account.to_csv(QC / "STUDY_FIT_ACCOUNTING.csv", index=False)

    keys = ["dgp", "template_id", "generator", "learner", "regime", "strategy", "delta", "policy", "confidence"]
    cell = pd.DataFrame(cell_metric_rows(long, keys))
    cell.to_parquet(SIM / "STUDY_CELL_DECISION_METRICS.parquet", index=False)
    macro = macro_from_cells(cell, ["dgp", "strategy", "delta", "policy", "confidence"])
    macro.to_csv(SIM / "STUDY_STRATEGY_SUMMARY.csv", index=False)
    macro.to_parquet(SIM / "STUDY_STRATEGY_SUMMARY.parquet", index=False)
    risk = macro[macro.policy == "RISK_YIELD"].copy()
    risk.to_csv(SIM / "STUDY_RISK_YIELD.csv", index=False)
    risk.to_parquet(SIM / "STUDY_RISK_YIELD.parquet", index=False)
    contrasts = make_contrasts(long)
    contrasts.to_csv(SIM / "STUDY_STRATEGY_CONTRASTS.csv", index=False)
    contrasts.to_parquet(SIM / "STUDY_STRATEGY_CONTRASTS.parquet", index=False)
    common = four_way_summary(long)
    common.to_csv(SIM / "STUDY_FOUR_WAY_COMMON_SUMMARY.csv", index=False)
    coverage, recovery = coverage_outputs(fits)
    coverage.to_csv(SIM / "STUDY_COVERAGE_CALIBRATION.csv", index=False)
    coverage.to_parquet(SIM / "STUDY_COVERAGE_CALIBRATION.parquet", index=False)
    recovery.to_csv(SIM / "STUDY_VARIANCE_RECOVERY.csv", index=False)
    recovery.to_parquet(SIM / "STUDY_VARIANCE_RECOVERY.parquet", index=False)

    qc = {
        "status": "PASS",
        "attempted_datasets": 72000,
        "attempted_fits": int(fit_account.attempted_n.sum()),
        "successful_fits": int(fit_account.successful_n.sum()),
        "failed_fits": int(fit_account.failed_n.sum()),
        "fit_accounting_rows": len(fit_account),
        "fit_accounting_expected_rows": 3 * 6 * 4 * 4,
        "estimation_instability_cells": int(fit_account.estimation_instability.sum()),
        "simulation_long_rows": len(long),
        "macro_cells_expected_per_stratum": 24,
        "strategy_summary_rows": len(macro),
        "contrast_rows": len(contrasts),
        "four_way_common_rows": len(common),
    }
    if qc["attempted_fits"] != expected_fits or qc["fit_accounting_rows"] != qc["fit_accounting_expected_rows"]:
        qc["status"] = "FAIL"
    (QC / "STUDY_SUMMARY_QC.json").write_text(json.dumps(qc, indent=2) + "\n")
    if qc["status"] != "PASS":
        raise RuntimeError(qc)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
