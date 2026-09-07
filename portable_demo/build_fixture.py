#!/usr/bin/env python3
"""Regenerate the fully fabricated 8x12 demo inputs without clinical data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "example"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seeds = [f"FSEED_{index:02d}" for index in range(1, 9)]
    sites = [f"FSITE_{index:02d}" for index in range(1, 13)]
    seed_effects = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    site_effects = [-0.030, -0.024, -0.018, -0.012, -0.006, -0.002,
                    0.002, 0.006, 0.012, 0.018, 0.024, 0.030]
    cells = []
    for site_index, (site, site_effect) in enumerate(zip(sites, site_effects)):
        for seed_index, (seed, seed_effect) in enumerate(zip(seeds, seed_effects)):
            interaction = 0.020 * (((seed_index + 2 * site_index) % 5) - 2)
            cells.append({
                "hospital_id": site,
                "seed_id": seed,
                "EUL": f"{0.120 + seed_effect + site_effect + interaction:.12f}",
            })
    write_csv(OUTPUT / "fixture_cells.csv", ["hospital_id", "seed_id", "EUL"], cells)

    covariance = []
    for site_index, site in enumerate(sites):
        diagonal = 0.000225 + site_index * 0.000005
        off_diagonal = diagonal * 0.20
        for row_seed in seeds:
            for column_seed in seeds:
                covariance.append({
                    "hospital_id": site,
                    "row_seed_id": row_seed,
                    "column_seed_id": column_seed,
                    "covariance": f"{diagonal if row_seed == column_seed else off_diagonal:.12f}",
                })
    write_csv(
        OUTPUT / "fixture_sampling_covariance.csv",
        ["hospital_id", "row_seed_id", "column_seed_id", "covariance"],
        covariance,
    )
    reliability = {
        "generator_id": "FABRICATED_GENERATOR_V1",
        "learner_id": "FABRICATED_LEARNER_V1",
        "generation_attempted_n": 8,
        "utility_estimable_n": 8,
        "generation_failure_n": 0,
        "source_result_version": "FABRICATED_8_SEED_12_HOSPITAL_FIXTURE_V1",
        "fixture_provenance": {
            "FIXTURE_CONTAINS_REAL_CLINICAL_DATA": "NO",
            "FIXTURE_DERIVED_FROM_REAL_PATIENT_ROWS": "NO",
            "FIXTURE_DERIVED_FROM_EMPIRICAL_HOSPITAL_ESTIMATES": "NO",
            "FIXTURE_PUBLIC_RELEASE_SAFE": "YES",
        },
    }
    (OUTPUT / "fixture_reliability.json").write_text(
        json.dumps(reliability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
