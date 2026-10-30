#!/usr/bin/env python3
"""Verify released E1 aggregate identities and frozen decision translation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_translation import aggregate_realization_decisions, p95_state


ROOT = Path(__file__).resolve().parents[2]
SUPP = ROOT / "supplementary_data"
EXPECTED = {
    "Supplementary_Data_01_E1A_REAL_TRAINED_BASELINE_AUROC.csv": "849d96486177d342b824edbd1e37e1d52bbc7fc5e83ad42f329695aa27fe3f57",
    "Supplementary_Data_02_E1B_REALIZATION_EUL_SUMMARY.csv": "d7ea650d21d06bcaa5205313a492aa6c884cb0664422355b5c577601fb24ec93",
    "Supplementary_Data_03_E1C_EMPIRICAL_M1_M2_M3_P95_DECISIONS.csv": "8b2780b45c4096df83c2a9f72ff2b3088f97ea5415f006df410084a3f329b953",
    "Supplementary_Data_04_E1D_M4_REALIZATION_DECISIONS.csv": "a4b195ea3b4567f16b2f614cd090cb3f417ddd660388d55991cb3627e1cfb115",
    "Supplementary_Data_05_E1E_H100_H200_H300_M1_TARGETB_DECISIONS.csv": "a14dab68110986cce26773075f2780022aa6225f746954c275ea02dd1579d4a7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for name, expected in EXPECTED.items():
        observed = sha256(SUPP / name)
        assert observed == expected, (name, observed, expected)

    for name in (
        "Supplementary_Data_03_E1C_EMPIRICAL_M1_M2_M3_P95_DECISIONS.csv",
        "Supplementary_Data_04_E1D_M4_REALIZATION_DECISIONS.csv",
        "Supplementary_Data_05_E1E_H100_H200_H300_M1_TARGETB_DECISIONS.csv",
    ):
        frame = pd.read_csv(SUPP / name)
        translated = frame["estimated_probability"].map(p95_state)
        assert translated.equals(frame["decision"]), name

    realization = pd.read_csv(SUPP / "Supplementary_Data_04_E1D_M4_REALIZATION_DECISIONS.csv")
    observed = aggregate_realization_decisions(realization)
    expected_counts = pd.read_csv(ROOT / "r2_source_data/E1D_M4_DECISION_COUNTS.csv")
    common = [c for c in observed.columns if c in expected_counts.columns]
    expected_counts = expected_counts[common].sort_values(common[:5]).reset_index(drop=True)
    observed = observed[common].sort_values(common[:5]).reset_index(drop=True)
    for column in common:
        if pd.api.types.is_numeric_dtype(observed[column]):
            assert np.allclose(observed[column], expected_counts[column], rtol=0, atol=5e-16), column
        else:
            assert observed[column].equals(expected_counts[column]), column

    print(json.dumps({
        "status": "PASS",
        "authoritative_export_hashes_verified": len(EXPECTED),
        "p95_rows_verified": int(sum(len(pd.read_csv(SUPP / n)) for n in list(EXPECTED)[2:])),
        "m4_count_rows_reproduced": len(observed),
        "clinical_data_access": False,
        "source_level_e1_regeneration": "REQUIRES_SEPARATELY_AUTHORIZED_CLINICAL_AND_RESTRICTED_INPUTS",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
