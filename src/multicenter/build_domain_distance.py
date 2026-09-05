#!/usr/bin/env python3
"""Build canonical R1 D_W using only predictor columns before outcome access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/postreview_upgrade/u1h_b12_protocol_conformant_repair_r1"
OUT = BASE / "domain_distance"
SCRIPT = ROOT / "scripts/postreview_upgrade/u1h_b12_protocol_conformant_repair_r1/01_build_dw_preoutcome.py"
SOURCE = ROOT / "outputs/preflight2b0/qualification/mimic_generator_contract_train.parquet"
EICU = ROOT / "outputs/postreview_upgrade/u0_r1_density_correction/eicu_replication_contract_density_corrected_v1.parquet"
FILL = ROOT / "outputs/preflight2b0/qualification/fill_values_mimic_train_median.json"

VARS = [
    "heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temperature_c",
    "wbc", "hemoglobin", "platelet_count", "creatinine", "bun_urea", "sodium", "potassium",
    "glucose", "bicarbonate", "chloride", "lactate", "ph_arterial", "pao2", "paco2",
]
FEATURES = [f"{v}_median" for v in VARS]


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    preflight_path=OUT/"DW_SOURCE_SCALE_PREFLIGHT.csv"
    table_path=OUT/"u1h_b12_r1_DW_all_hospitals.csv"
    lock_path=OUT/"DW_PREOUTCOME_LOCK.json"
    for p in (preflight_path,table_path,lock_path):
        if p.exists(): raise FileExistsError(p)

    source=pd.read_parquet(SOURCE,columns=FEATURES)
    eicu=pd.read_parquet(EICU,columns=["hospitalid",*FEATURES]).rename(columns={"hospitalid":"hospital_id"})
    fill=json.loads(FILL.read_text())
    for c in FEATURES:
        if source[c].isna().any():
            raise RuntimeError(f"Frozen source predictor unexpectedly missing: {c}")
        if eicu[c].isna().any():
            if c not in fill: raise RuntimeError(f"No frozen fill value: {c}")
            eicu[c]=eicu[c].fillna(float(fill[c]))
    if eicu.hospital_id.isna().any(): raise RuntimeError("Missing hospital_id")

    eps=np.finfo(np.float64).eps
    rows=[]; scales={}
    for c in FEATURES:
        x=source[c].to_numpy(np.float64)
        finite=x[np.isfinite(x)]
        n=len(finite); mu=float(np.mean(finite)) if n else float("nan")
        sd=float(np.std(finite,ddof=1)) if n>=2 else float("nan")
        tau=float(np.sqrt(eps)*max(1.0,abs(mu))) if np.isfinite(mu) else float("nan")
        if n<2: status="FAIL_FEWER_THAN_TWO_VALID"
        elif not np.isfinite(sd): status="FAIL_NONFINITE_SCALE"
        elif sd==0: status="FAIL_ZERO_SCALE"
        elif sd<=tau: status="FAIL_NEAR_ZERO_SCALE"
        else: status="PASS"
        rows.append({"variable":c,"source_N":n,"source_mean":mu,"source_SD_ddof1":sd,"near_zero_threshold":tau,"scale_status":status})
        scales[c]=sd
    pre=pd.DataFrame(rows)
    pre.to_csv(preflight_path,index=False)
    failures=pre.query("scale_status != 'PASS'")
    if len(pre)!=21 or not failures.empty:
        raise SystemExit("BLOCKED_SOURCE_SCALE_DEGENERACY")

    out=[]; centered_max=0.0
    groups=eicu.groupby("hospital_id",sort=True,observed=True)
    for hid,dh in groups:
        row={"hospital_id":int(hid),"N_predictor_rows":int(len(dh))}
        vals=[]
        for v,c in zip(VARS,FEATURES,strict=True):
            xs=source[c].to_numpy(np.float64); xh=dh[c].to_numpy(np.float64); sd=scales[c]; mu=float(xs.mean())
            if not np.isfinite(xh).all(): raise RuntimeError(f"Nonfinite hospital predictor {hid}/{c}")
            d=float(wasserstein_distance(xs,xh)/sd)
            dz=float(wasserstein_distance((xs-mu)/sd,(xh-mu)/sd))
            centered_max=max(centered_max,abs(d-dz))
            row[f"distance_{v}"]=d; vals.append(d)
        row["n_valid_DW_components"]=len(vals)
        row["D_W"]=float(np.mean(vals))
        out.append(row)
    table=pd.DataFrame(out)
    if len(table)!=208 or not (table.n_valid_DW_components==21).all(): raise RuntimeError("D_W universe/component failure")
    table.to_csv(table_path,index=False)
    lock={
        "method_document":"docs/ESTIMANDS.md",
        "source_domain":"REAL_MIMIC_TRAIN",
        "canonical_protocol_location":"docs/ESTIMANDS.md",
        "feature_count":21,"median_only":True,"scale":"SAMPLE_SD","ddof":1,
        "age_in_DW":False,"sex_in_DW":False,"component_exclusions":0,
        "outcome_columns_accessed":0,"prediction_columns_accessed":0,"performance_columns_accessed":0,
        "hospitals_characterized":int(len(table)),
        "source_scale_min":float(pre.source_SD_ddof1.min()),"source_scale_max":float(pre.source_SD_ddof1.max()),
        "near_zero_failures":0,
        "centered_vs_direct_max_abs_difference":centered_max,
        "DW_table_path":str(table_path.relative_to(ROOT)),"DW_table_sha256":sha(table_path),
        "DW_script_path":str(SCRIPT.relative_to(ROOT)),"DW_script_sha256":sha(SCRIPT),
        "source_artifact_sha256":sha(SOURCE),"eicu_artifact_sha256":sha(EICU),"fill_registry_sha256":sha(FILL),
    }
    lock_path.write_text(json.dumps(lock,indent=2)+"\n")
    print(json.dumps({"scale_pass":21,"hospitals":len(table),"DW_sha256":lock["DW_table_sha256"],"centered_max_abs":centered_max},indent=2))


if __name__=="__main__": main()
