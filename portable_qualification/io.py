"""Portable CSV/JSON input and output helpers."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_source_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_summary(path: Path) -> dict[str, Any]:
    """Read one post-fit summary from JSON or a one-row CSV file."""
    if path.suffix.lower() == ".json":
        return read_json(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError("summary CSV must contain exactly one data row")
        return rows[0]
    raise ValueError("summary must be a .json or .csv file")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_card(card: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in card.items():
        if isinstance(value, (list, dict)):
            result[key] = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        elif value is None:
            result[key] = ""
        else:
            result[key] = value
    return result


def read_cells(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"hospital_id", "seed_id", "EUL"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("cells CSV requires hospital_id, seed_id, and EUL")
    return rows


def read_covariance(path: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"hospital_id", "row_seed_id", "column_seed_id", "covariance"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("covariance CSV has missing required columns")
    result: dict[str, tuple[list[str], np.ndarray]] = {}
    for hospital in sorted({row["hospital_id"] for row in rows}):
        block = [row for row in rows if row["hospital_id"] == hospital]
        seeds = sorted({row["row_seed_id"] for row in block} | {row["column_seed_id"] for row in block})
        index = {seed: position for position, seed in enumerate(seeds)}
        matrix = np.full((len(seeds), len(seeds)), np.nan)
        for row in block:
            i = index[row["row_seed_id"]]
            j = index[row["column_seed_id"]]
            if np.isfinite(matrix[i, j]):
                raise ValueError(f"duplicate covariance entry for {hospital}/{i}/{j}")
            matrix[i, j] = float(row["covariance"])
        if np.isnan(matrix).any():
            raise ValueError(f"incomplete covariance block for {hospital}")
        result[hospital] = (seeds, matrix)
    return result
