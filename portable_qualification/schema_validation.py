"""JSON Schema and semantic validation for portable evidence cards."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jsonschema

from .constants import (
    PROHIBITED_CARD_FIELDS,
    QUALIFICATION_BOUNDARY,
    QUALIFICATION_MODEL,
    U4_CALIBRATION_BOUNDARY,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "evidence_card.schema.json"


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_numeric_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_numeric_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_numeric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def validate_evidence_card(card: dict[str, Any]) -> None:
    """Fail closed on schema or cross-field semantic violations."""
    schema = jsonschema.validators.validator_for({"$schema": "https://json-schema.org/draft/2020-12/schema"})
    schema.check_schema(__import__("json").loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    validator = schema(__import__("json").loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    validator.validate(card)
    prohibited = {key.lower() for key in _walk_keys(card)} & PROHIBITED_CARD_FIELDS
    if prohibited:
        raise ValueError(f"prohibited evidence-card field(s): {sorted(prohibited)}")
    if not all(math.isfinite(value) for value in _walk_numeric_values(card)):
        raise ValueError("all evidence-card numeric values must be finite")
    attempted = card["generation_attempted_n"]
    estimable = card["utility_estimable_n"]
    failure = card["generation_failure_n"]
    if estimable + failure != attempted:
        raise ValueError("generation reliability count identity failed")
    if not math.isclose(card["generation_failure_rate"], failure / attempted, abs_tol=1e-12):
        raise ValueError("generation failure rate identity failed")
    grid = card["realization_grid"]
    if grid != sorted(set(grid)) or any(m < 1 for m in grid):
        raise ValueError("realization_grid must contain unique increasing positive integers")
    expected_keys = {str(m) for m in grid}
    for field in ("V_het_by_m", "V_pred_by_m", "G_by_m"):
        if set(card[field]) != expected_keys:
            raise ValueError(f"{field} keys must match realization_grid")
    probabilities = card["model_based_tolerance_probability_by_m"]
    if card["user_tolerance_delta"] is None:
        if probabilities is not None:
            raise ValueError("tolerance probabilities must be null when delta is absent")
    elif probabilities is None or set(probabilities) != expected_keys:
        raise ValueError("tolerance probability keys must match realization_grid")
    if card["qualification_model"] != QUALIFICATION_MODEL:
        raise ValueError("qualification model identifier drift")
    if card["qualification_boundary"] != QUALIFICATION_BOUNDARY:
        raise ValueError("qualification boundary drift")
    if card["u4_calibration_boundary"] != U4_CALIBRATION_BOUNDARY:
        raise ValueError("U4 calibration boundary drift")
    if card["target_A_PI_lower"] > card["target_A_PI_upper"]:
        raise ValueError("Target A interval endpoints are reversed")
    if card["target_B_PI_lower"] > card["target_B_PI_upper"]:
        raise ValueError("Target B interval endpoints are reversed")
