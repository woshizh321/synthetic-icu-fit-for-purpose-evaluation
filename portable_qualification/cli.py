"""Command-line interface for portable post-fit qualification."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .algebra import qualify_summary
from .constants import (
    INTERMEDIATE_M_BOUNDARY,
    M_EQ_BOUNDARY,
    M_EQ_DEFINITION,
    M_EQ_LABEL,
    U4_CALIBRATION_BOUNDARY,
)
from .crossed_reml import fit_crossed_reml
from .io import (
    combined_source_sha,
    flatten_card,
    read_cells,
    read_covariance,
    read_json,
    read_summary,
    write_csv,
    write_json,
)
from .schema_validation import validate_evidence_card


def _fit_summary(cells_path: Path, covariance_path: Path, reliability_path: Path) -> dict:
    reliability = read_json(reliability_path)
    fitted = fit_crossed_reml(read_cells(cells_path), read_covariance(covariance_path))
    if not fitted["converged"]:
        raise RuntimeError(f"crossed REML did not converge: {fitted['optimizer_message']}")
    mean = fitted["mu_eul"]
    se = fitted["se_mu"]
    seed = fitted["sigma2_seed"]
    hospital = fitted["sigma2_hospital"]
    interaction = fitted["sigma2_seed_hospital"]
    target_a_scale = math.sqrt(se * se + hospital)
    target_b_scale = math.sqrt(se * se + seed + hospital + interaction)
    source_sha = combined_source_sha([cells_path, covariance_path, reliability_path])
    return {
        **fitted,
        "generator_id": str(reliability["generator_id"]),
        "learner_id": str(reliability["learner_id"]),
        "generation_attempted_n": int(reliability["generation_attempted_n"]),
        "utility_estimable_n": int(reliability["utility_estimable_n"]),
        "generation_failure_n": int(reliability["generation_failure_n"]),
        "target_A_PI_lower": mean - 1.96 * target_a_scale,
        "target_A_PI_upper": mean + 1.96 * target_a_scale,
        "target_B_PI_lower": mean - 1.96 * target_b_scale,
        "target_B_PI_upper": mean + 1.96 * target_b_scale,
        "source_result_version": str(reliability["source_result_version"]),
        "source_result_sha": source_sha,
        "input_hashes": {
            "fixture_cells.csv": __import__("hashlib").sha256(cells_path.read_bytes()).hexdigest(),
            "fixture_sampling_covariance.csv": __import__("hashlib").sha256(covariance_path.read_bytes()).hexdigest(),
            "fixture_reliability.json": __import__("hashlib").sha256(reliability_path.read_bytes()).hexdigest(),
        },
    }


def _write_qualification(summary: dict, output_dir: Path, m_max: int, delta: float | None) -> dict:
    card, rows = qualify_summary(summary, m_max=m_max, delta=delta)
    validate_evidence_card(card)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evidence_card.json", card)
    write_csv(output_dir / "evidence_card.csv", [flatten_card(card)])
    write_csv(output_dir / "realization_budget.csv", rows)
    metadata = {
        "u4_calibration_boundary": U4_CALIBRATION_BOUNDARY,
        "variance_equivalence_label": M_EQ_LABEL,
        "variance_equivalence_definition": M_EQ_DEFINITION,
        "variance_equivalence_boundary": M_EQ_BOUNDARY,
        "intermediate_m_boundary": INTERMEDIATE_M_BOUNDARY,
        "output_files": ["evidence_card.json", "evidence_card.csv", "realization_budget.csv"],
    }
    write_json(output_dir / "qualification_metadata.json", metadata)
    return card


def command_fit(args: argparse.Namespace) -> int:
    summary = _fit_summary(args.cells, args.covariance, args.reliability)
    write_json(args.output, summary)
    print(json.dumps({"fit_summary": str(args.output), "converged": True}, sort_keys=True))
    return 0


def command_qualify(args: argparse.Namespace) -> int:
    summary = read_summary(args.summary)
    card = _write_qualification(summary, args.output_dir, args.m_max, args.delta)
    print(
        json.dumps(
            {
                "evidence_card": str(args.output_dir / "evidence_card.json"),
                "realization_budget": str(args.output_dir / "realization_budget.csv"),
                "model_based_tolerance_probability_calculated": args.delta is not None,
                "m_eq": card["m_eq"],
                "m_eq_label": M_EQ_LABEL,
                "u4_calibration_boundary": U4_CALIBRATION_BOUNDARY,
            },
            sort_keys=True,
        )
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    fixture = args.fixture
    cells = fixture / "fixture_cells.csv"
    covariance = fixture / "fixture_sampling_covariance.csv"
    reliability = fixture / "fixture_reliability.json"
    for path in (cells, covariance, reliability):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _fit_summary(cells, covariance, reliability)
    write_json(args.output_dir / "fit_summary.json", summary)
    card = _write_qualification(summary, args.output_dir, args.m_max, args.delta)
    print(
        json.dumps(
            {
                "demo_output": str(args.output_dir),
                "fit_converged": True,
                "model_based_tolerance_probability_calculated": args.delta is not None,
                "m_eq": card["m_eq"],
                "u4_calibration_boundary": U4_CALIBRATION_BOUNDARY,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portable_qualification",
        description="CPU-only post-fit qualification for crossed synthetic-data summaries.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit the existing crossed model to aggregate cells")
    fit_parser.add_argument("--cells", required=True, type=Path)
    fit_parser.add_argument("--covariance", required=True, type=Path)
    fit_parser.add_argument("--reliability", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    fit_parser.set_defaults(func=command_fit)

    qualify_parser = subparsers.add_parser("qualify", help="translate an aggregate fit summary")
    qualify_parser.add_argument("--summary", required=True, type=Path)
    qualify_parser.add_argument("--m-max", type=int, default=20)
    qualify_parser.add_argument("--delta", type=float, default=None)
    qualify_parser.add_argument("--output-dir", type=Path, default=Path("portable_outputs"))
    qualify_parser.set_defaults(func=command_qualify)

    demo_parser = subparsers.add_parser("demo", help="run the complete fabricated fixture workflow")
    demo_parser.add_argument("--fixture", required=True, type=Path)
    demo_parser.add_argument("--m-max", type=int, default=20)
    demo_parser.add_argument("--delta", type=float, default=None)
    demo_parser.add_argument("--output-dir", type=Path, default=Path("portable_demo_output"))
    demo_parser.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # CLI fails closed with a concise technical error.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
