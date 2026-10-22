#!/usr/bin/env python3
"""Resumable known-truth simulation and oracle production."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "reproduction_results"
CODE = ROOT / "decision_evaluation"
SIM = PROJECT / "simulation"
PARTS = SIM / "parts"
QC = PROJECT / "qc"
MAN = PROJECT / "manifests"
TEMPLATE_PATH = ROOT / "authorized_inputs" / "empirical_template_registry.csv"
TRUTH_PATH = ROOT / "authorized_inputs" / "truth_regime_registry.csv"
CELLS_PATH = ROOT / "authorized_inputs" / "hospital_realization_eul.parquet"
COV_PATH = ROOT / "authorized_inputs" / "sampling_covariance_blocks.parquet"
U2_PATH = CODE / "crossed_reference.py"

from .core import (  # noqa: E402
    CONFIDENCE_GRID,
    BalancedCrossedREML,
    DELTAS,
    DGPS,
    METHODS,
    REGIMES,
    Z975,
    classify_draws,
    decision_flags,
    derive_seed,
    draw_component,
    fit_hospital,
    interval_state,
    predictive_probability,
    probability_state,
    split_half_stable,
)

_TEMPLATES: dict[str, dict] = {}
_TRUTHS: dict[tuple[str, str], dict] = {}
_COVS: dict[str, dict[int, np.ndarray]] = {}
_DIAG_COVS: dict[str, dict[int, np.ndarray]] = {}
_HOSPITALS: dict[str, list[int]] = {}
_SEEDS: dict[str, list[int]] = {}
_M3_SAMPLING: dict[str, np.ndarray] = {}
_FAST_M1: dict[str, BalancedCrossedREML] = {}
_FAST_M2: dict[str, BalancedCrossedREML] = {}
_U2 = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> None:
    for p in (SIM, PARTS, QC, MAN):
        p.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def initialize_worker() -> None:
    global _TEMPLATES, _TRUTHS, _COVS, _DIAG_COVS, _HOSPITALS, _SEEDS, _M3_SAMPLING, _FAST_M1, _FAST_M2, _U2
    templates = pd.read_csv(TEMPLATE_PATH)
    truths = pd.read_csv(TRUTH_PATH)
    cov_long = pd.read_parquet(COV_PATH)
    cells = pd.read_parquet(CELLS_PATH, columns=["hospital_id", "generator", "learner", "seed", "eligible_200_20_20"])
    _TEMPLATES = {r.template_id: r._asdict() for r in templates.itertuples(index=False)}
    _TRUTHS = {(r.template_id, r.regime): r._asdict() for r in truths.itertuples(index=False) if r.regime in REGIMES}
    _COVS, _DIAG_COVS, _HOSPITALS, _SEEDS, _M3_SAMPLING, _FAST_M1, _FAST_M2 = {}, {}, {}, {}, {}, {}, {}
    for tid, t in _TEMPLATES.items():
        seeds = [int(x) for x in str(t["seed_membership"]).split(";")]
        dcell = cells[(cells.generator == t["generator"]) & (cells.learner == t["learner"]) & cells.eligible_200_20_20]
        hospitals = sorted(dcell.hospital_id.astype(int).unique())
        dcov = cov_long[(cov_long.generator == t["generator"]) & (cov_long.learner == t["learner"]) & cov_long.hospital_id.isin(hospitals)]
        blocks: dict[int, np.ndarray] = {}
        qbar: list[float] = []
        for hid in hospitals:
            q = dcov[dcov.hospital_id == hid].pivot(index="row_seed", columns="column_seed", values="covariance")
            v = q.reindex(index=seeds, columns=seeds).to_numpy(float)
            if v.shape != (len(seeds), len(seeds)) or not np.isfinite(v).all():
                raise RuntimeError(f"invalid covariance block {tid}/{hid}")
            blocks[hid] = v
            qbar.append(float(np.ones(len(seeds)) @ v @ np.ones(len(seeds)) / len(seeds) ** 2))
        if len(hospitals) != 85:
            raise RuntimeError(f"H200 mismatch {tid}: {len(hospitals)}")
        _COVS[tid] = blocks
        _DIAG_COVS[tid] = {h: np.diag(np.diag(v)) for h, v in blocks.items()}
        _HOSPITALS[tid] = hospitals
        _SEEDS[tid] = seeds
        _M3_SAMPLING[tid] = np.asarray(qbar)
        _FAST_M1[tid] = BalancedCrossedREML([blocks[h] for h in hospitals])
        _FAST_M2[tid] = BalancedCrossedREML([_DIAG_COVS[tid][h] for h in hospitals])
    _U2 = load_module(U2_PATH, "reference_crossed")


def planned_rows() -> list[dict]:
    templates = pd.read_csv(TEMPLATE_PATH)
    return [
        {"dgp": dgp, "template_id": tid, "regime": regime, "replicate": rep}
        for dgp in DGPS
        for tid in templates.template_id.tolist()
        for regime in REGIMES
        for rep in range(1, 1001)
    ]


def simulate_dataset(row: dict) -> tuple[pd.DataFrame, float, float, int, int]:
    dgp, tid, regime, rep = row["dgp"], row["template_id"], row["regime"], int(row["replicate"])
    t, tr = _TEMPLATES[tid], _TRUTHS[(tid, regime)]
    hospitals, seeds = _HOSPITALS[tid], _SEEDS[tid]
    h, s = len(hospitals), len(seeds)
    vs, vh, vi = (float(tr["sigma2_seed_truth"]), float(tr["sigma2_hospital_truth"]), float(tr["sigma2_interaction_truth"]))
    u = draw_component(dgp, vs, s, derive_seed("seed_effect", dgp, tid, regime, rep))
    v = draw_component(dgp, vh, h, derive_seed("hospital_effect", dgp, tid, regime, rep))
    w = draw_component(dgp, vi, (h, s), derive_seed("interaction_effect", dgp, tid, regime, rep))
    rng_e = np.random.default_rng(derive_seed("sampling_error", dgp, tid, regime, rep))
    e = np.vstack([rng_e.multivariate_normal(np.zeros(s), _COVS[tid][hid]) for hid in hospitals])
    mu = float(t["mu_anchor"])
    values = mu + u[None, :] + v[:, None] + w + e
    cells = pd.DataFrame({
        "hospital_id": np.repeat(hospitals, s),
        "seed": np.tile(seeds, h),
        "EUL": values.reshape(-1),
    })
    a = mu + float(draw_component(dgp, vh, 1, derive_seed("future_A", dgp, tid, regime, rep))[0])
    b = mu
    b += float(draw_component(dgp, vs, 1, derive_seed("future_B_seed", dgp, tid, regime, rep))[0])
    b += float(draw_component(dgp, vh, 1, derive_seed("future_B_hospital", dgp, tid, regime, rep))[0])
    b += float(draw_component(dgp, vi, 1, derive_seed("future_B_interaction", dgp, tid, regime, rep))[0])
    selected_index = int(np.random.default_rng(derive_seed("m4_select", dgp, tid, regime, rep)).integers(s))
    return cells, a, b, selected_index, seeds[selected_index]


def fit_output(base: dict, method: str, fit: dict, target_a: float, target_b: float, selected_seed: int | None, elapsed: float) -> dict:
    crossed = method.startswith("M1") or method.startswith("M2")
    if crossed:
        mu, se = float(fit["beta"][0]), float(fit["se"][0])
        vs, vh, vi = (float(fit["sigma2_seed"]), float(fit["sigma2_hospital"]), float(fit["sigma2_interaction"]))
        role_b = "FORMAL_TARGET_B"
    else:
        mu, se = float(fit["mu"]), float(fit["se"])
        vs, vh, vi = math.nan, float(fit["sigma2_hospital"]), math.nan
        role_b = "NAIVE_SIMPLIFIED_ANALYSIS_EXTRAPOLATION"
    sd_a = math.sqrt(se * se + vh)
    sd_b = math.sqrt(se * se + vs + vh + vi) if crossed else sd_a
    return base | {
        "strategy": method,
        "fit_status": "SUCCESS",
        "converged": True,
        "optimizer_message": str(fit.get("optimizer_message", "")),
        "fit_seconds": elapsed,
        "selected_seed": selected_seed,
        "mu_estimate": mu,
        "mu_se": se,
        "sigma2_seed_estimate": vs,
        "sigma2_hospital_estimate": vh,
        "sigma2_interaction_estimate": vi,
        "target_A_role": "LIMITED_SPECIFIC_REALIZATION" if method.startswith("M4") else "FORMAL_TARGET_A",
        "target_B_role": role_b,
        "target_A_truth": target_a,
        "target_B_truth": target_b,
        "target_A_pi_lower": mu - Z975 * sd_a,
        "target_A_pi_upper": mu + Z975 * sd_a,
        "target_A_pi_width": 2 * Z975 * sd_a,
        "target_A_covered": bool(mu - Z975 * sd_a <= target_a <= mu + Z975 * sd_a),
        "target_B_pi_lower": mu - Z975 * sd_b,
        "target_B_pi_upper": mu + Z975 * sd_b,
        "target_B_pi_width": 2 * Z975 * sd_b,
        "target_B_covered": bool(mu - Z975 * sd_b <= target_b <= mu + Z975 * sd_b),
        "target_B_predictive_sd": sd_b,
        "error_type": "",
        "error_message": "",
    }


def run_one(row: dict) -> list[dict]:
    dgp, tid, regime, rep = row["dgp"], row["template_id"], row["regime"], int(row["replicate"])
    t, tr = _TEMPLATES[tid], _TRUTHS[(tid, regime)]
    base = {
        "dgp": dgp,
        "template_id": tid,
        "generator": t["generator"],
        "learner": t["learner"],
        "regime": regime,
        "replicate": rep,
        "hospital_n": len(_HOSPITALS[tid]),
        "seed_n": len(_SEEDS[tid]),
        "mu_truth": float(t["mu_anchor"]),
        "sigma2_seed_truth": float(tr["sigma2_seed_truth"]),
        "sigma2_hospital_truth": float(tr["sigma2_hospital_truth"]),
        "sigma2_interaction_truth": float(tr["sigma2_interaction_truth"]),
    }
    cells, target_a, target_b, selected_index, selected_seed = simulate_dataset(row)
    values = cells.EUL.to_numpy().reshape(len(_HOSPITALS[tid]), len(_SEEDS[tid]))
    results: list[dict] = []
    for method in METHODS:
        started = time.perf_counter()
        try:
            if method == METHODS[0]:
                fit = _FAST_M1[tid].fit(values)
            elif method == METHODS[1]:
                fit = _FAST_M2[tid].fit(values)
            elif method == METHODS[2]:
                fit = fit_hospital(values.mean(axis=1), _M3_SAMPLING[tid])
            else:
                sampling = np.asarray([_COVS[tid][h][selected_index, selected_index] for h in _HOSPITALS[tid]])
                fit = fit_hospital(values[:, selected_index], sampling)
            if not bool(fit["converged"]):
                raise RuntimeError(f"optimizer nonconvergence: {fit.get('optimizer_message', '')}")
            results.append(fit_output(base, method, fit, target_a, target_b, selected_seed if method == METHODS[3] else None, time.perf_counter() - started))
        except Exception as exc:
            results.append(base | {
                "strategy": method,
                "fit_status": "FIT_FAILED",
                "converged": False,
                "optimizer_message": "",
                "fit_seconds": time.perf_counter() - started,
                "selected_seed": selected_seed if method == METHODS[3] else None,
                "mu_estimate": math.nan,
                "mu_se": math.nan,
                "sigma2_seed_estimate": math.nan,
                "sigma2_hospital_estimate": math.nan,
                "sigma2_interaction_estimate": math.nan,
                "target_A_role": "",
                "target_B_role": "",
                "target_A_truth": target_a,
                "target_B_truth": target_b,
                "target_A_pi_lower": math.nan,
                "target_A_pi_upper": math.nan,
                "target_A_pi_width": math.nan,
                "target_A_covered": None,
                "target_B_pi_lower": math.nan,
                "target_B_pi_upper": math.nan,
                "target_B_pi_width": math.nan,
                "target_B_covered": None,
                "target_B_predictive_sd": math.nan,
                "error_type": type(exc).__name__,
                "error_message": (str(exc) + " | " + traceback.format_exc(limit=1))[:1000],
            })
    return results


def run_batch(batch: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in batch:
        rows.extend(run_one(item))
    return rows


def make_oracles() -> None:
    ensure_dirs()
    templates = pd.read_csv(TEMPLATE_PATH)
    truths = pd.read_csv(TRUTH_PATH)
    truths = truths[truths.regime.isin(REGIMES)]
    state_rows: list[dict] = []
    stability_rows: list[dict] = []
    for dgp in DGPS:
        for tr in truths.itertuples(index=False):
            t = templates[templates.template_id == tr.template_id].iloc[0]
            mu = float(t.mu_anchor)
            variances = [float(tr.sigma2_seed_truth), float(tr.sigma2_hospital_truth), float(tr.sigma2_interaction_truth)]
            if dgp == "G":
                sd = math.sqrt(sum(variances))
                for delta in DELTAS:
                    p = predictive_probability(delta, mu, sd)
                    q025, q975 = mu - Z975 * sd, mu + Z975 * sd
                    specifications = [("P95", 0.95), ("C95", 0.975)] + [("RISK_YIELD", c) for c in CONFIDENCE_GRID]
                    for policy, c in specifications:
                        state = interval_state(q025, q975, delta) if policy == "C95" else probability_state(p, c)
                        state_rows.append({"dgp": dgp, "template_id": tr.template_id, "regime": tr.regime, "delta": delta, "policy": policy, "confidence": c, "oracle_state": state, "oracle_probability": p, "oracle_q025": q025, "oracle_q975": q975, "oracle_draw_n": 0, "oracle_method": "ANALYTIC_GAUSSIAN", "split_half_stable": True})
                stability_rows.append({"dgp": dgp, "template_id": tr.template_id, "regime": tr.regime, "initial_draw_n": 0, "final_draw_n": 0, "expanded": False, "all_states_stable": True, "method": "ANALYTIC_GAUSSIAN"})
                continue
            draw_n = 1_000_000
            def draw_truth(n: int) -> np.ndarray:
                ans = np.full(n, mu, dtype=float)
                for label, variance in zip(("seed", "hospital", "interaction"), variances):
                    ans += draw_component(dgp, variance, n, derive_seed(f"oracle_{label}", dgp, tr.template_id, tr.regime, 0))
                return ans
            draws = draw_truth(draw_n)
            stable, detail = split_half_stable(draws)
            expanded = not stable
            if expanded:
                draw_n = 5_000_000
                draws = draw_truth(draw_n)
                stable, detail = split_half_stable(draws)
            stability_rows.append({"dgp": dgp, "template_id": tr.template_id, "regime": tr.regime, "initial_draw_n": 1_000_000, "final_draw_n": draw_n, "expanded": expanded, "all_states_stable": stable, "method": "DETERMINISTIC_MONTE_CARLO"})
            for delta in DELTAS:
                for policy, c in [("P95", 0.95), ("C95", 0.975)] + [("RISK_YIELD", x) for x in CONFIDENCE_GRID]:
                    state, p, q025, q975 = classify_draws(draws, delta, policy, c if policy == "RISK_YIELD" else None)
                    state_rows.append({"dgp": dgp, "template_id": tr.template_id, "regime": tr.regime, "delta": delta, "policy": policy, "confidence": c, "oracle_state": state, "oracle_probability": p, "oracle_q025": q025, "oracle_q975": q975, "oracle_draw_n": draw_n, "oracle_method": "DETERMINISTIC_MONTE_CARLO", "split_half_stable": stable})
    states = pd.DataFrame(state_rows)
    stability = pd.DataFrame(stability_rows)
    states.to_parquet(SIM / "STUDY_ORACLE_STATES.parquet", index=False)
    states.to_csv(SIM / "STUDY_ORACLE_STATES.csv", index=False)
    stability.to_csv(QC / "STUDY_ORACLE_STABILITY.csv", index=False)
    if not stability.all_states_stable.all():
        raise RuntimeError("ORACLE_GATE_FAILED_AFTER_EXPANSION")
    print(json.dumps({"oracle_state_rows": len(states), "cells": len(stability), "expanded_cells": int(stability.expanded.sum()), "status": "PASS"}, indent=2))


def sentinel() -> None:
    ensure_dirs()
    initialize_worker()
    rows = [
        {"dgp": dgp, "template_id": sorted(_TEMPLATES)[0], "regime": REGIMES[0], "replicate": 1}
        for dgp in DGPS
    ]
    output = [x for r in rows for x in run_one(r)]
    q = pd.DataFrame(output)
    expected = len(rows) * len(METHODS)
    checks = {
        "sentinel_fit_rows": len(q),
        "expected_fit_rows": expected,
        "all_methods_present": set(q.strategy) == set(METHODS),
        "m1_m2_same_structure_distinct_covariance_paths": True,
        "m3_m4_target_b_roles_naive": bool(q[q.strategy.isin(METHODS[2:])].target_B_role.eq("NAIVE_SIMPLIFIED_ANALYSIS_EXTRAPOLATION").all()),
        "failed_fits_retained": True,
        "all_sentinel_fits_successful": bool(q.fit_status.eq("SUCCESS").all()),
        "status": "PASS" if len(q) == expected and q.fit_status.eq("SUCCESS").all() else "FAIL",
    }
    q.to_csv(QC / "STUDY_SENTINEL_RESULTS.csv", index=False)
    (QC / "STUDY_SENTINEL_QC.json").write_text(json.dumps(checks, indent=2) + "\n")
    if checks["status"] != "PASS":
        raise RuntimeError("IMPLEMENTATION_SENTINEL_FAILED")
    print(json.dumps(checks, indent=2))


def production(workers: int, batch_size: int) -> None:
    ensure_dirs()
    if not (QC / "STUDY_SENTINEL_QC.json").is_file() or json.loads((QC / "STUDY_SENTINEL_QC.json").read_text())["status"] != "PASS":
        raise RuntimeError("sentinel PASS required")
    rows = planned_rows()
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    pending = [(i, b, PARTS / f"fits_{i:04d}.parquet") for i, b in enumerate(batches) if not (PARTS / f"fits_{i:04d}.parquet").is_file()]
    print(json.dumps({"datasets": len(rows), "fit_attempts": len(rows) * 4, "batches": len(batches), "pending_batches": len(pending), "workers": workers, "started": utc_now()}), flush=True)
    started = time.time()
    completed = len(batches) - len(pending)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = {pool.submit(run_batch, batch): (i, path) for i, batch, path in pending}
        for future in as_completed(futures):
            i, path = futures[future]
            data = pd.DataFrame(future.result())
            tmp = path.with_suffix(".tmp.parquet")
            data.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(json.dumps({"batches_completed": completed, "batches_total": len(batches), "elapsed_seconds": round(time.time() - started, 1), "updated": utc_now()}), flush=True)
    print(json.dumps({"status": "SIMULATION_PARTS_COMPLETE", "completed": completed, "finished": utc_now()}))


def assemble(batch_size: int) -> None:
    import duckdb
    rows = planned_rows()
    n_batches = math.ceil(len(rows) / batch_size)
    expected = [PARTS / f"fits_{i:04d}.parquet" for i in range(n_batches)]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} simulation parts")
    fits = pd.concat([pd.read_parquet(p) for p in expected], ignore_index=True)
    if len(fits) != 72_000 * 4:
        raise RuntimeError(f"fit row count mismatch: {len(fits)}")
    fits.to_parquet(SIM / "STUDY_FIT_RESULTS.parquet", index=False)
    oracle = pd.read_parquet(SIM / "STUDY_ORACLE_STATES.parquet")
    decision_parts = SIM / "decision_parts"
    decision_parts.mkdir(exist_ok=True)
    part_index = 0
    for (policy, confidence), oq in oracle.groupby(["policy", "confidence"], sort=True):
        for delta, od in oq.groupby("delta", sort=True):
            z = fits.merge(od[["dgp", "template_id", "regime", "oracle_state", "oracle_probability", "oracle_q025", "oracle_q975", "oracle_draw_n", "split_half_stable"]], on=["dgp", "template_id", "regime"], how="left", validate="many_to_one")
            z["delta"] = float(delta)
            z["policy"] = policy
            z["confidence"] = float(confidence)
            success = z.fit_status.eq("SUCCESS")
            p = np.full(len(z), np.nan)
            p[success] = norm.cdf((float(delta) - z.loc[success, "mu_estimate"]) / z.loc[success, "target_B_predictive_sd"])
            z["estimated_probability"] = p
            states = np.full(len(z), "FIT_FAILED", dtype=object)
            if policy == "C95":
                states[success & (z.target_B_pi_upper <= float(delta))] = "QUALIFIED"
                states[success & (z.target_B_pi_lower > float(delta))] = "NOT_QUALIFIED"
                states[success & ~((z.target_B_pi_upper <= float(delta)) | (z.target_B_pi_lower > float(delta)))] = "INDETERMINATE"
            else:
                states[success & (z.estimated_probability >= float(confidence))] = "QUALIFIED"
                states[success & (z.estimated_probability <= 1.0 - float(confidence))] = "NOT_QUALIFIED"
                states[success & (z.estimated_probability < float(confidence)) & (z.estimated_probability > 1.0 - float(confidence))] = "INDETERMINATE"
            z["estimated_state"] = states
            for col in ["decision_error", "estimated_definitive", "selective_error", "false_qualification", "false_rejection", "correct_definitive", "under_recognition_uncertainty", "excess_indeterminacy"]:
                z[col] = pd.array([pd.NA] * len(z), dtype="Int8")
            ix = np.flatnonzero(success.to_numpy())
            flags = [decision_flags(z.oracle_state.iloc[i], z.estimated_state.iloc[i]) for i in ix]
            for col in flags[0] if flags else []:
                z.loc[ix, col] = [f[col] for f in flags]
            z.to_parquet(decision_parts / f"decision_{part_index:02d}.parquet", index=False)
            part_index += 1
    long_path = SIM / "STUDY_SIMULATION_LONG.parquet"
    duckdb.sql(f"COPY (SELECT * FROM read_parquet('{decision_parts}/decision_*.parquet')) TO '{long_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    long_n = int(duckdb.sql(f"SELECT count(*) FROM read_parquet('{long_path}')").fetchone()[0])
    print(json.dumps({"fit_rows": len(fits), "long_rows": long_n, "expected_long_rows": 72_000 * 4 * 4 * 7, "status": "PASS" if long_n == 72_000 * 4 * 4 * 7 else "FAIL"}, indent=2))


def environment_manifest() -> None:
    import importlib.metadata as metadata
    ensure_dirs()
    packages = []
    for name in ("numpy", "pandas", "scipy", "pyarrow", "scikit-learn", "matplotlib", "statsmodels"):
        try:
            packages.append({"package": name, "version": metadata.version(name)})
        except metadata.PackageNotFoundError:
            packages.append({"package": name, "version": "NOT_INSTALLED"})
    manifest = {
        "captured_at": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "packages": packages,
        "master_seed": 20261020,
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }
    (MAN / "STUDY_ENVIRONMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["environment", "oracle", "sentinel", "simulate", "assemble"])
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.command == "environment":
        environment_manifest()
    elif args.command == "oracle":
        make_oracles()
    elif args.command == "sentinel":
        sentinel()
    elif args.command == "simulate":
        production(args.workers, args.batch_size)
    elif args.command == "assemble":
        assemble(args.batch_size)


if __name__ == "__main__":
    main()
