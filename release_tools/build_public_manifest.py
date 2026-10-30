#!/usr/bin/env python3
"""Build the versioned public-tree manifest after the privacy gate."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PUBLIC_RELEASE_MANIFEST_v1.4.0.csv"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def classify(path: str) -> tuple[str, str]:
    if path.startswith("extension/e3_e5/public_inputs/"):
        return "SIMULATED_REPLICATE_REPRODUCTION_INPUT", "original V2 historical locked simulation outputs projected to the frozen P95 replay contract"
    if path.startswith("supplementary_data/") or path.startswith("r2_source_data/") or path.startswith("manuscript_source_data/"):
        return "APPROVED_AGGREGATE_OR_SIMULATION_OUTPUT", "locked manuscript or extension authority"
    if path.startswith("r2_manuscript_figures/") or path.startswith("manuscript_figures/") or path.startswith("publication_assets/"):
        return "PUBLICATION_DISPLAY", "final display or supplement authority"
    if path.endswith(".py"):
        return "PUBLIC_SOFTWARE", "public executed or verification software"
    return "PUBLIC_CONFIG_METADATA_OR_DOCUMENTATION", "public software and release metadata"


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        asset_class, authority = classify(rel)
        rows.append({
            "path": rel,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "asset_class": asset_class,
            "source_authority": authority,
            "contains_patient_level_data": "false",
            "contains_restricted_derivative": "false",
            "public_release_status": "APPROVED",
            "exclusion_reason": "",
            "notes": "",
        })
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT.name}: {len(rows)} files")


if __name__ == "__main__":
    main()
