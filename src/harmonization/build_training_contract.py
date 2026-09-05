"""PREFLIGHT-2B0 / B0-06,07: build the frozen MIMIC generator-contract training
table (DEC-007 schema) from the locked MIMIC-train split only, plus a small
deterministic qualification subset shared identically across all candidates.
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

ROOT = os.environ.get("ICU_PROJECT_ROOT", ".")
STG2A = f"{ROOT}/outputs/preflight2a/staging"
OUT = f"{ROOT}/outputs/preflight2b0"
QUAL = f"{OUT}/qualification"
os.makedirs(QUAL, exist_ok=True)
os.makedirs(f"{OUT}/tables", exist_ok=True)

VARS = ["heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temperature_c",
        "wbc", "hemoglobin", "platelet_count", "creatinine", "bun_urea", "sodium", "potassium",
        "glucose", "bicarbonate", "chloride", "lactate", "ph_arterial", "pao2", "paco2"]
A_STATS = ["first", "last", "min", "max", "median"]

mA = pd.read_parquet(f"{STG2A}/repA_mimic.parquet").set_index("stay_id")

# reproduce the EXACT locked P2A-06 80/20 stratified split (same df order, same seed)
FEAT_PROXY = ["age", "sex_male"] + [f"{v}__{s}" for v in VARS for s in A_STATS]
Xtr, Xte, ytr, yte = train_test_split(
    mA[FEAT_PROXY], mA["y"], test_size=0.20, stratify=mA["y"], random_state=SEED)
train_idx = Xtr.index

mtr = mA.loc[train_idx].copy()
assert len(mtr) == 40745, f"MIMIC train split mismatch: got {len(mtr)}, expected 40745 (DEC-005)"

# ---- build the DEC-007 150-field contract ----
contract = pd.DataFrame(index=mtr.index)
contract["age"] = mtr["age"].astype(float)
contract["sex_male"] = mtr["sex_male"].astype(int)
for v in VARS:
    for s in A_STATS:
        contract[f"{v}_{s}"] = mtr[f"{v}__{s}"].astype(float)
    contract[f"{v}_observed_any"] = (1 - mtr[f"{v}__missing"]).astype(int)
    contract[f"{v}_observed_hour_density"] = mtr[f"{v}__density"].astype(float)
contract["mortality_7d_post_landmark"] = mtr["y"].astype(int)

# DEC-008: fill missing value-stat fields with MIMIC-TRAIN-ONLY median (observed rows only)
fill_values = {}
for v in VARS:
    obs_mask = contract[f"{v}_observed_any"] == 1
    for s in A_STATS:
        col = f"{v}_{s}"
        med = contract.loc[obs_mask, col].median()
        fill_values[col] = float(med)
        contract[col] = contract[col].fillna(med)

n_fields = contract.shape[1]
assert n_fields == 150, f"contract field count mismatch: got {n_fields}, expected 150"

contract.reset_index().to_parquet(f"{QUAL}/mimic_generator_contract_train.parquet")

with open(f"{QUAL}/fill_values_mimic_train_median.json", "w") as fh:
    json.dump(fill_values, fh, indent=2)

# ---- manifest ----
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "source": "outputs/preflight2a/staging/repA_mimic.parquet (P2A-06 Representation A extract)",
    "split_provenance": "sklearn.train_test_split(test_size=0.20, stratify=y, random_state=42) "
                        "on the DEC-003 primary certainty-based MIMIC cohort (N=50,932); "
                        "TRAIN partition only used here (SICdb and MIMIC-test never touched)",
    "n_rows": int(len(contract)),
    "n_fields": int(n_fields),
    "field_schema": {c: str(contract[c].dtype) for c in contract.columns},
    "outcome_field": "mortality_7d_post_landmark",
    "outcome_prevalence_pct": round(100 * contract["mortality_7d_post_landmark"].mean(), 4),
    "outcome_n_events": int(contract["mortality_7d_post_landmark"].sum()),
    "missingness_prevalence_pct_by_variable": {
        v: round(100 * (1 - contract[f"{v}_observed_any"]).mean(), 3) for v in VARS
    },
    "file_sha256": file_sha256(f"{QUAL}/mimic_generator_contract_train.parquet"),
    "seed": SEED,
    "dec_references": ["DEC-003", "DEC-005", "DEC-006", "DEC-007", "DEC-008"],
    "note": "This is a reproducible implementation-qualification artifact, NOT the final "
            "publication dataset by default.",
}
with open(f"{QUAL}/manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

# ---- B0-07: deterministic qualification subset, shared identically across all generators ----
QUAL_N = 3000
rng = np.random.RandomState(SEED)
qual_idx = rng.choice(contract.index.values, size=QUAL_N, replace=False)
qual_idx.sort()
qual_subset = contract.loc[qual_idx].copy()
qual_subset.reset_index().to_parquet(f"{QUAL}/qualification_subset_n3000_seed42.parquet")

qual_manifest = {
    "n_rows": int(len(qual_subset)),
    "n_fields": int(qual_subset.shape[1]),
    "sampling_rule": "numpy.random.RandomState(42).choice(train_index, size=3000, replace=False), "
                     "sorted ascending for determinism",
    "pct_of_full_train": round(100 * QUAL_N / len(contract), 2),
    "outcome_prevalence_pct": round(100 * qual_subset["mortality_7d_post_landmark"].mean(), 4),
    "outcome_n_events": int(qual_subset["mortality_7d_post_landmark"].sum()),
    "identical_subset_shared_across_all_generators": True,
    "file_sha256": file_sha256(f"{QUAL}/qualification_subset_n3000_seed42.parquet"),
}
with open(f"{QUAL}/qualification_subset_manifest.json", "w") as fh:
    json.dump(qual_manifest, fh, indent=2)

print(json.dumps({"train_manifest": manifest["n_rows"], "n_fields": n_fields,
                  "outcome_prevalence_pct": manifest["outcome_prevalence_pct"],
                  "qual_subset": qual_manifest}, indent=2, default=str))
print("DONE")
