"""Frozen U5 post-fit qualification algebra."""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING
from typing import Any

from .constants import (
    INTERMEDIATE_M_STATUS,
    QUALIFICATION_BOUNDARY,
    QUALIFICATION_MODEL,
    SCHEMA_VERSION,
    SOFTWARE_VERSION,
    U4_CALIBRATION_BOUNDARY,
)


def _finite_nonnegative(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def variance_equivalence_count(
    sigma2_seed: float,
    sigma2_hospital: float,
    sigma2_seed_hospital: float,
) -> tuple[int | str, str]:
    """Return the frozen point-estimate variance-equivalence count and status."""
    seed = _finite_nonnegative("sigma2_seed", sigma2_seed)
    hospital = _finite_nonnegative("sigma2_hospital", sigma2_hospital)
    interaction = _finite_nonnegative("sigma2_seed_hospital", sigma2_seed_hospital)
    generator_related = seed + interaction
    if hospital == 0 and generator_related == 0:
        return 1, "NO_TRUE_HETEROGENEITY"
    if hospital == 0:
        return "INF", "HOSPITAL_VARIANCE_ZERO_GENERATOR_VARIANCE_POSITIVE"
    if generator_related == 0:
        return 1, "NO_GENERATOR_RELATED_HETEROGENEITY"
    # Decimal-from-string prevents binary representation noise at an exact
    # integer boundary from increasing the mathematical ceiling by one.
    ratio = (Decimal(str(seed)) + Decimal(str(interaction))) / Decimal(str(hospital))
    count = int(ratio.to_integral_value(rounding=ROUND_CEILING))
    return max(1, count), "FINITE_POINT_ESTIMATE"


def target_a_variance(se_mu: float, sigma2_hospital: float) -> float:
    se = _finite_nonnegative("se_mu", se_mu)
    hospital = _finite_nonnegative("sigma2_hospital", sigma2_hospital)
    return se * se + hospital


def target_b_variance(
    se_mu: float,
    sigma2_seed: float,
    sigma2_hospital: float,
    sigma2_seed_hospital: float,
) -> float:
    se = _finite_nonnegative("se_mu", se_mu)
    seed = _finite_nonnegative("sigma2_seed", sigma2_seed)
    hospital = _finite_nonnegative("sigma2_hospital", sigma2_hospital)
    interaction = _finite_nonnegative("sigma2_seed_hospital", sigma2_seed_hospital)
    return se * se + seed + hospital + interaction


def realization_quantities(
    m: int,
    se_mu: float,
    sigma2_seed: float,
    sigma2_hospital: float,
    sigma2_seed_hospital: float,
) -> dict[str, float | int | str]:
    """Calculate the locked variance quantities for one integer realization count."""
    count = _positive_integer("m", m)
    se = _finite_nonnegative("se_mu", se_mu)
    seed = _finite_nonnegative("sigma2_seed", sigma2_seed)
    hospital = _finite_nonnegative("sigma2_hospital", sigma2_hospital)
    interaction = _finite_nonnegative("sigma2_seed_hospital", sigma2_seed_hospital)
    raw = seed + interaction
    after = raw / count
    v_het = hospital + after
    v_pred = se * se + v_het
    g_value = 0.0 if v_het == 0 else after / v_het
    return {
        "m": count,
        "generator_related_variance_raw": raw,
        "generator_related_variance_after_m": after,
        "V_het_m": v_het,
        "V_pred_m": v_pred,
        "G_m": g_value,
        "intermediate_m_status": INTERMEDIATE_M_STATUS,
    }


def model_based_tolerance_probability(
    delta: float,
    mu: float,
    v_pred: float,
) -> float:
    """Calculate Phi((delta-mu)/sqrt(V_pred)); delta must be user supplied."""
    tolerance = _finite("delta", delta)
    mean = _finite("mu", mu)
    variance = _finite_nonnegative("v_pred", v_pred)
    if variance == 0:
        return 1.0 if mean <= tolerance else 0.0
    z_value = (tolerance - mean) / math.sqrt(variance)
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


def _counts(summary: dict[str, Any]) -> tuple[int, int, int, float]:
    attempted = _positive_integer("generation_attempted_n", summary["generation_attempted_n"])
    estimable = int(summary["utility_estimable_n"])
    failure = int(summary.get("generation_failure_n", attempted - estimable))
    if estimable < 0 or failure < 0 or estimable + failure != attempted:
        raise ValueError("generation reliability counts are inconsistent")
    return attempted, estimable, failure, failure / attempted


def qualify_summary(
    summary: dict[str, Any],
    *,
    m_max: int = 20,
    delta: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Translate one post-fit aggregate summary into an evidence card and long table."""
    maximum = _positive_integer("m_max", m_max)
    attempted, estimable, failure, failure_rate = _counts(summary)
    mean = _finite("mu_eul", summary["mu_eul"])
    se = _finite_nonnegative("se_mu", summary["se_mu"])
    seed = _finite_nonnegative("sigma2_seed", summary["sigma2_seed"])
    hospital = _finite_nonnegative("sigma2_hospital", summary["sigma2_hospital"])
    interaction = _finite_nonnegative("sigma2_seed_hospital", summary["sigma2_seed_hospital"])
    supplied_delta = None if delta is None else _finite("delta", delta)
    m_eq, m_eq_status = variance_equivalence_count(seed, hospital, interaction)
    rows = [realization_quantities(m, se, seed, hospital, interaction) for m in range(1, maximum + 1)]
    keys = [str(row["m"]) for row in rows]
    tolerance = None
    if supplied_delta is not None:
        tolerance = {
            key: model_based_tolerance_probability(supplied_delta, mean, float(row["V_pred_m"]))
            for key, row in zip(keys, rows)
        }
    source_sha = str(summary["source_result_sha"])
    if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_result_sha must be a lowercase SHA256")
    card = {
        "schema_version": SCHEMA_VERSION,
        "generator_id": str(summary["generator_id"]),
        "learner_id": str(summary["learner_id"]),
        "generation_attempted_n": attempted,
        "utility_estimable_n": estimable,
        "generation_failure_n": failure,
        "generation_failure_rate": failure_rate,
        "mu_eul": mean,
        "se_mu": se,
        "sigma2_seed": seed,
        "sigma2_hospital": hospital,
        "sigma2_seed_hospital": interaction,
        "target_A_PI_lower": _finite("target_A_PI_lower", summary["target_A_PI_lower"]),
        "target_A_PI_upper": _finite("target_A_PI_upper", summary["target_A_PI_upper"]),
        "target_B_PI_lower": _finite("target_B_PI_lower", summary["target_B_PI_lower"]),
        "target_B_PI_upper": _finite("target_B_PI_upper", summary["target_B_PI_upper"]),
        "m_eq": m_eq,
        "m_eq_status": m_eq_status,
        "realization_grid": list(range(1, maximum + 1)),
        "V_het_by_m": {key: float(row["V_het_m"]) for key, row in zip(keys, rows)},
        "V_pred_by_m": {key: float(row["V_pred_m"]) for key, row in zip(keys, rows)},
        "G_by_m": {key: float(row["G_m"]) for key, row in zip(keys, rows)},
        "user_tolerance_delta": supplied_delta,
        "model_based_tolerance_probability_by_m": tolerance,
        "qualification_model": QUALIFICATION_MODEL,
        "qualification_boundary": QUALIFICATION_BOUNDARY,
        "u4_calibration_boundary": U4_CALIBRATION_BOUNDARY,
        "source_result_version": str(summary["source_result_version"]),
        "source_result_sha": source_sha,
        "software_version": SOFTWARE_VERSION,
    }
    for row in rows:
        row.update({"generator": card["generator_id"], "learner": card["learner_id"]})
        row["m_eq"] = m_eq
        row["m_eq_status"] = m_eq_status
        row["model_based_tolerance_probability"] = (
            None if tolerance is None else tolerance[str(row["m"])]
        )
    return card, rows
