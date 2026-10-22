#!/usr/bin/env python3
"""Exploratory secondary generation-success × conditional-utility analysis."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "reproduction_results"
CODE = ROOT / "decision_evaluation"
OUT = PROJECT / "operational"
SOURCE = ROOT / "authorized_inputs" / "generation_evidence_cards.json"
EMPIRICAL = PROJECT / "empirical_hospital/STUDY_H100_H200_H300_MODELS.parquet"

from .core import DELTAS, predictive_probability  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards = json.loads(SOURCE.read_text())
    if isinstance(cards, dict):
        cards = cards.get("evidence_cards", cards.get("cards", cards))
    if isinstance(cards, dict):
        cards = list(cards.values())
    reliability = {}
    for x in cards:
        generator = x.get("generator_id", x.get("generator"))
        attempted = int(x["generation_attempted_n"])
        estimable = int(x["utility_estimable_n"])
        key = (generator, attempted, estimable)
        reliability[key] = True
    by_generator = {}
    for generator, attempted, estimable in reliability:
        by_generator.setdefault(generator, set()).add((attempted, estimable))
    if any(len(v) != 1 for v in by_generator.values()):
        raise RuntimeError(f"learner-dependent generation counts: {by_generator}")

    empirical = pd.read_parquet(EMPIRICAL)
    empirical = empirical[empirical.hospital_population == "H200"]
    rows = []
    for generator, values in sorted(by_generator.items()):
        attempted, estimable = next(iter(values))
        p_success = estimable / attempted
        ci = binomtest(estimable, attempted).proportion_ci(confidence_level=.95, method="exact")
        for erow in empirical[empirical.generator == generator].itertuples(index=False):
            sd = (float(erow.mu_SE) ** 2 + float(erow.sigma2_seed) + float(erow.sigma2_hospital) + float(erow.sigma2_interaction)) ** .5
            for delta in DELTAS:
                conditional = predictive_probability(delta, float(erow.mu_EUL), sd)
                rows.append({
                    "analysis_status": "EXPLORATORY_SECONDARY",
                    "generator": generator,
                    "learner": erow.learner,
                    "hospital_population": "H200",
                    "target": "Target_B",
                    "delta": delta,
                    "generation_attempted_n": attempted,
                    "utility_estimable_n": estimable,
                    "generation_failure_n": attempted - estimable,
                    "p_success": p_success,
                    "p_success_exact_95ci_lower": float(ci.low),
                    "p_success_exact_95ci_upper": float(ci.high),
                    "p_target_B_le_delta_given_success": conditional,
                    "P_operational": p_success * conditional,
                    "decomposition": "P(success) * P(Target-B EUL <= delta | success)",
                    "independence_assumption": False,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "STUDY_OPERATIONAL_SUITABILITY.csv", index=False)
    out.to_parquet(OUT / "STUDY_OPERATIONAL_SUITABILITY.parquet", index=False)
    summary = out[["generator", "generation_attempted_n", "utility_estimable_n", "generation_failure_n", "p_success", "p_success_exact_95ci_lower", "p_success_exact_95ci_upper"]].drop_duplicates()
    summary.to_csv(OUT / "STUDY_GENERATION_SUCCESS.csv", index=False)
    qc = {
        "status": "PASS" if len(out) == 24 else "FAIL",
        "generator_n": out.generator.nunique(),
        "rows": len(out),
        "expected_rows": 3 * 2 * 4,
        "all_labeled_exploratory_secondary": bool(out.analysis_status.eq("EXPLORATORY_SECONDARY").all()),
        "source": str(SOURCE),
    }
    (OUT / "STUDY_OPERATIONAL_QC.json").write_text(json.dumps(qc, indent=2) + "\n")
    if qc["status"] != "PASS":
        raise RuntimeError(qc)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
