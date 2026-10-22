#!/usr/bin/env python3
"""Conditional-on-V_h parametric bootstrap for six H200 M1 models."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "reproduction_results"
CODE = ROOT / "decision_evaluation"
OUT = PROJECT / "variance_uncertainty"
PARTS = OUT / "parts"
QC = PROJECT / "qc"
TEMPLATE_PATH = ROOT / "authorized_inputs" / "empirical_template_registry.csv"
CELLS_PATH = ROOT / "authorized_inputs" / "hospital_realization_eul.parquet"
COV_PATH = ROOT / "authorized_inputs" / "sampling_covariance_blocks.parquet"
U2_PATH = CODE / "crossed_reference.py"
EMPIRICAL_PATH = PROJECT / "empirical_hospital/STUDY_H100_H200_H300_MODELS.parquet"

from .core import BalancedCrossedREML, Z975, derive_seed, variance_equivalence_count  # noqa: E402

_T: dict[str, dict] = {}
_COV: dict[str, dict[int, np.ndarray]] = {}
_H: dict[str, list[int]] = {}
_S: dict[str, list[int]] = {}
_BASE: dict[str, dict] = {}
_FAST: dict[str, BalancedCrossedREML] = {}
_U2 = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def initialize_worker() -> None:
    global _T, _COV, _H, _S, _BASE, _FAST, _U2
    templates = pd.read_csv(TEMPLATE_PATH)
    cells = pd.read_parquet(CELLS_PATH)
    covlong = pd.read_parquet(COV_PATH)
    empirical = pd.read_parquet(EMPIRICAL_PATH)
    empirical = empirical[empirical.hospital_population == "H200"]
    _T = {r.template_id: r._asdict() for r in templates.itertuples(index=False)}
    _COV, _H, _S, _BASE, _FAST = {}, {}, {}, {}, {}
    for t in templates.itertuples(index=False):
        seeds = [int(x) for x in str(t.seed_membership).split(";")]
        d = cells[(cells.generator == t.generator) & (cells.learner == t.learner) & cells.eligible_200_20_20 & cells.seed.isin(seeds)]
        hospitals = sorted(d.hospital_id.astype(int).unique())
        dc = covlong[(covlong.generator == t.generator) & (covlong.learner == t.learner) & covlong.hospital_id.isin(hospitals)]
        blocks = {}
        for hid in hospitals:
            q = dc[dc.hospital_id == hid].pivot(index="row_seed", columns="column_seed", values="covariance")
            blocks[hid] = q.reindex(index=seeds, columns=seeds).to_numpy(float)
        erow = empirical[empirical.template_id == t.template_id].iloc[0]
        _COV[t.template_id] = blocks
        _H[t.template_id] = hospitals
        _S[t.template_id] = seeds
        _BASE[t.template_id] = {
            "mu": float(erow.mu_EUL),
            "sigma2_seed": float(erow.sigma2_seed),
            "sigma2_hospital": float(erow.sigma2_hospital),
            "sigma2_interaction": float(erow.sigma2_interaction),
        }
        _FAST[t.template_id] = BalancedCrossedREML([blocks[h] for h in hospitals])
    _U2 = load_module(U2_PATH, "reference_crossed")


def run_one(job: dict) -> tuple[dict, list[dict]]:
    tid, rep = job["template_id"], int(job["replicate"])
    t, b, hospitals, seeds = _T[tid], _BASE[tid], _H[tid], _S[tid]
    h, s = len(hospitals), len(seeds)
    rng_u = np.random.default_rng(derive_seed("bootstrap_seed_effect", "EMPIRICAL", tid, "H200", rep))
    rng_v = np.random.default_rng(derive_seed("bootstrap_hospital_effect", "EMPIRICAL", tid, "H200", rep))
    rng_w = np.random.default_rng(derive_seed("bootstrap_interaction_effect", "EMPIRICAL", tid, "H200", rep))
    rng_e = np.random.default_rng(derive_seed("bootstrap_sampling_error", "EMPIRICAL", tid, "H200", rep))
    u = rng_u.normal(0, math.sqrt(b["sigma2_seed"]), s)
    v = rng_v.normal(0, math.sqrt(b["sigma2_hospital"]), h)
    w = rng_w.normal(0, math.sqrt(b["sigma2_interaction"]), (h, s))
    e = np.vstack([rng_e.multivariate_normal(np.zeros(s), _COV[tid][hid]) for hid in hospitals])
    y = b["mu"] + u[None, :] + v[:, None] + w + e
    cells = pd.DataFrame({"hospital_id": np.repeat(hospitals, s), "seed": np.tile(seeds, h), "EUL": y.reshape(-1)})
    base = {"template_id": tid, "generator": t["generator"], "learner": t["learner"], "replicate": rep, "conditional_on_V_h": True}
    try:
        fit = _FAST[tid].fit(y)
        if not fit["converged"]:
            raise RuntimeError(fit["optimizer_message"])
        mu, se = float(fit["beta"][0]), float(fit["se"][0])
        vs, vh, vi = (float(fit["sigma2_seed"]), float(fit["sigma2_hospital"]), float(fit["sigma2_interaction"]))
        total = vs + vh + vi
        m_eq, m_status = variance_equivalence_count(vs, vh, vi)
        boundary = min(vs, vh, vi) <= math.exp(-27.63) * 1.01
        row = base | {
            "fit_status": "SUCCESS",
            "optimizer_message": str(fit["optimizer_message"]),
            "mu_EUL": mu,
            "mu_SE": se,
            "sigma2_seed": vs,
            "sigma2_hospital": vh,
            "sigma2_interaction": vi,
            "share_seed": vs / total,
            "share_hospital": vh / total,
            "share_interaction": vi / total,
            "boundary_flag": boundary,
            "m_eq": m_eq,
            "m_eq_status": m_status,
            "error_type": "",
            "error_message": "",
        }
        rng_a = np.random.default_rng(derive_seed("bootstrap_future_A", "EMPIRICAL", tid, "H200", rep))
        rng_b = np.random.default_rng(derive_seed("bootstrap_future_B", "EMPIRICAL", tid, "H200", rep))
        a = mu + rng_a.normal(0, math.sqrt(vh), 200)
        bb = mu + rng_b.normal(0, math.sqrt(total), 200)
        draws = [
            {"template_id": tid, "replicate": rep, "target": target, "draw_index": i + 1, "latent_value": float(value)}
            for target, values in (("A", a), ("B", bb))
            for i, value in enumerate(values)
        ]
        return row, draws
    except Exception as exc:
        return base | {
            "fit_status": "FIT_FAILED",
            "optimizer_message": "",
            "mu_EUL": math.nan,
            "mu_SE": math.nan,
            "sigma2_seed": math.nan,
            "sigma2_hospital": math.nan,
            "sigma2_interaction": math.nan,
            "share_seed": math.nan,
            "share_hospital": math.nan,
            "share_interaction": math.nan,
            "boundary_flag": False,
            "m_eq": None,
            "m_eq_status": "FIT_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        }, []


def run_batch(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    fits, draws = [], []
    for job in batch:
        fit, dd = run_one(job)
        fits.append(fit)
        draws.extend(dd)
    return fits, draws


def produce(workers: int, batch_size: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    templates = pd.read_csv(TEMPLATE_PATH)
    jobs = [{"template_id": tid, "replicate": rep} for tid in templates.template_id for rep in range(1, 1001)]
    batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]
    pending = [(i, b) for i, b in enumerate(batches) if not (PARTS / f"fit_{i:04d}.parquet").is_file()]
    print(json.dumps({"bootstrap_fits": len(jobs), "batches": len(batches), "pending": len(pending), "workers": workers, "started": utc_now()}), flush=True)
    completed = len(batches) - len(pending)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = {pool.submit(run_batch, b): i for i, b in pending}
        for future in as_completed(futures):
            i = futures[future]
            fits, draws = future.result()
            fp, dp = PARTS / f"fit_{i:04d}.parquet", PARTS / f"draw_{i:04d}.parquet"
            ft, dt = fp.with_suffix(".tmp.parquet"), dp.with_suffix(".tmp.parquet")
            pd.DataFrame(fits).to_parquet(ft, index=False)
            pd.DataFrame(draws).to_parquet(dt, index=False)
            os.replace(ft, fp)
            os.replace(dt, dp)
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(json.dumps({"batches_completed": completed, "batches_total": len(batches), "elapsed_seconds": round(time.time() - started, 1)}), flush=True)


def summarize(batch_size: int) -> None:
    templates = pd.read_csv(TEMPLATE_PATH)
    n_batches = math.ceil(6000 / batch_size)
    fps = [PARTS / f"fit_{i:04d}.parquet" for i in range(n_batches)]
    dps = [PARTS / f"draw_{i:04d}.parquet" for i in range(n_batches)]
    if any(not p.is_file() for p in fps + dps):
        raise RuntimeError("bootstrap parts incomplete")
    fits = pd.concat([pd.read_parquet(p) for p in fps], ignore_index=True)
    draws = pd.concat([pd.read_parquet(p) for p in dps], ignore_index=True)
    hospital_boundary = (fits.fit_status == "SUCCESS") & (fits.sigma2_hospital <= math.exp(-27.63) * 1.01)
    generator_positive = (fits.sigma2_seed.fillna(0) + fits.sigma2_interaction.fillna(0)) > math.exp(-27.63) * 1.01
    fits.loc[hospital_boundary & generator_positive, "m_eq"] = None
    fits.loc[hospital_boundary & generator_positive, "m_eq_status"] = "HOSPITAL_VARIANCE_AT_NUMERICAL_BOUNDARY_GENERATOR_VARIANCE_POSITIVE"
    fits.loc[hospital_boundary & ~generator_positive, "m_eq"] = 1
    fits.loc[hospital_boundary & ~generator_positive, "m_eq_status"] = "ALL_RELEVANT_VARIANCES_AT_NUMERICAL_BOUNDARY"
    fits.to_parquet(OUT / "STUDY_VARIANCE_BOOTSTRAP.parquet", index=False)
    fits.to_csv(OUT / "STUDY_VARIANCE_BOOTSTRAP.csv", index=False)
    summary_rows: list[dict] = []
    meq_rows: list[dict] = []
    for tid, d in fits.groupby("template_id", sort=True):
        ok = d[d.fit_status == "SUCCESS"]
        for variable in ("mu_EUL", "sigma2_seed", "sigma2_hospital", "sigma2_interaction", "share_seed", "share_hospital", "share_interaction"):
            x = ok[variable].astype(float)
            summary_rows.append({"template_id": tid, "variable": variable, "successful_n": len(ok), "q025": float(x.quantile(.025)), "median": float(x.median()), "q975": float(x.quantile(.975)), "conditional_on_V_h": True})
        finite = pd.to_numeric(ok.m_eq, errors="coerce").dropna()
        meq_rows.append({
            "template_id": tid,
            "successful_n": len(ok),
            "finite_n": len(finite),
            "unbounded_n": int(ok.m_eq_status.astype(str).str.contains("HOSPITAL_VARIANCE_AT_NUMERICAL_BOUNDARY").sum()),
            "boundary_n": int(ok.boundary_flag.sum()),
            "q025": float(finite.quantile(.025)) if len(finite) else math.nan,
            "median": float(finite.median()) if len(finite) else math.nan,
            "q975": float(finite.quantile(.975)) if len(finite) else math.nan,
            "status": "DESCRIPTIVE_MODEL_BASED_ONLY",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "STUDY_VARIANCE_BOOTSTRAP_INTERVALS.csv", index=False)
    pd.DataFrame(meq_rows).to_csv(OUT / "STUDY_M_EQ_BOOTSTRAP.csv", index=False)

    empirical = pd.read_parquet(EMPIRICAL_PATH)
    empirical = empirical[empirical.hospital_population == "H200"]
    interval_rows = []
    for erow in empirical.itertuples(index=False):
        for target, lo, hi in (("A", erow.target_A_lower, erow.target_A_upper), ("B", erow.target_B_lower, erow.target_B_upper)):
            q = draws[(draws.template_id == erow.template_id) & (draws.target == target)].latent_value
            interval_rows.append({
                "template_id": erow.template_id,
                "generator": erow.generator,
                "learner": erow.learner,
                "target": target,
                "plugin_lower": float(lo),
                "plugin_upper": float(hi),
                "bootstrap_mixture_lower": float(q.quantile(.025)),
                "bootstrap_mixture_upper": float(q.quantile(.975)),
                "successful_bootstrap_refits": int(fits[(fits.template_id == erow.template_id) & (fits.fit_status == "SUCCESS")].shape[0]),
                "draws_per_successful_refit": 200,
                "conditional_on_V_h": True,
            })
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(OUT / "STUDY_PREDICTION_INTERVAL_COMPARISON.csv", index=False)
    intervals.to_parquet(OUT / "STUDY_PREDICTION_INTERVAL_COMPARISON.parquet", index=False)
    account = fits.groupby("template_id").fit_status.agg(attempted_n="size", successful_n=lambda x: int((x == "SUCCESS").sum()), failed_n=lambda x: int((x != "SUCCESS").sum())).reset_index()
    account["failure_rate"] = account.failed_n / account.attempted_n
    account["estimation_instability"] = account.failure_rate > .05
    account.to_csv(OUT / "STUDY_BOOTSTRAP_FIT_ACCOUNTING.csv", index=False)
    qc = {
        "status": "PASS" if len(fits) == 6000 and int(account.attempted_n.sum()) == 6000 else "FAIL",
        "attempted": len(fits),
        "successful": int(fits.fit_status.eq("SUCCESS").sum()),
        "failed": int(fits.fit_status.ne("SUCCESS").sum()),
        "mixture_draw_rows": len(draws),
        "conditional_on_estimated_V_h": True,
        "V_h_uncertainty_propagated": False,
        "future_draws_per_successful_refit_per_target": 200,
    }
    (QC / "STUDY_BOOTSTRAP_QC.json").write_text(json.dumps(qc, indent=2) + "\n")
    if qc["status"] != "PASS":
        raise RuntimeError(qc)
    print(json.dumps(qc, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["produce", "summarize"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.command == "produce":
        produce(args.workers, args.batch_size)
    else:
        summarize(args.batch_size)


if __name__ == "__main__":
    main()
