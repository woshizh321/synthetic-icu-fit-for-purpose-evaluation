"""PREFLIGHT-2A / P2A-01,02,03: corrected feature extraction.

Builds, for MIMIC-IV and SICdb, Representation A (static 24h) and
Representation B (4x6h bins) feature matrices, each in three channels:
  V   = physiologic values only (first,last,min,max,median)
  VM  = V + missingness indicator
  VMD = VM + normalized observed-hour density (DEC-002)

Outcome: primary certainty-based 7-day post-landmark mortality (DEC-003).
No raw measurement counts anywhere (PREFLIGHT-1 defect corrected).
Diagnostic/protocol-characterization only. Deterministic, seed=42.
"""
import json
import os
import numpy as np
import pandas as pd
import duckdb

np.random.seed(42)

ROOT = os.environ.get("ICU_PROJECT_ROOT", ".")
OUT1 = f"{ROOT}/outputs/preflight1"
OUT2 = f"{ROOT}/outputs/preflight2a"
STG1 = f"{OUT1}/staging"
STG2 = f"{OUT2}/staging"
SIC = os.environ["SICDB_ROOT"]
os.makedirs(STG2, exist_ok=True)
os.makedirs(f"{OUT2}/logs", exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'; PRAGMA threads=10;")

# locked 21-variable map (identical to PREFLIGHT-1 common_variable_mapping.csv)
VARS = {
    "heart_rate":      ([220045], [707, 724, 708], (10, 300)),
    "sbp":             ([220050, 220179, 225309], [701, 704], (20, 300)),
    "dbp":             ([220051, 220180, 225310], [702, 705], (5, 225)),
    "map":             ([220052, 220181, 225312], [703, 706], (10, 250)),
    "respiratory_rate": ([220210, 224690], [719, 2274, 2280], (1, 80)),
    "spo2":            ([220277], [710], (20, 100)),
    "temperature_c":   ([223762, 223761], [709], (25, 45)),
    "wbc":             ([51301], [301], (0, 200)),
    "hemoglobin":      ([51222, 50811], [289, 288, 658], (2, 25)),
    "platelet_count":  ([51265], [314, 315], (1, 2000)),
    "creatinine":      ([50912], [367], (0.1, 30)),
    "bun_urea":        ([51006], [355], (1, 250)),
    "sodium":          ([50983, 50824], [469, 455, 686], (90, 200)),
    "potassium":       ([50971, 50822], [463, 453, 685], (1, 10)),
    "glucose":         ([50931, 50809], [348, 331, 656], (10, 2000)),
    "bicarbonate":     ([50882, 50803], [456, 451, 666], (2, 60)),
    "chloride":        ([50902, 50806], [450, 683], (50, 200)),
    "lactate":         ([50813], [465, 454, 657], (0.1, 30)),
    "ph_arterial":     ([50820], [538, 688, 663], (6.5, 8.0)),
    "pao2":            ([50821], [444, 689, 664], (10, 700)),
    "paco2":           ([50818], [443, 687, 665], (5, 200)),
}
MIMIC_VITALS = {"heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temperature_c"}
UREA_TO_BUN = 2.14

# ============================================================== MIMIC cohort/outcome (DEC-003)
print("[MIMIC] cohort + certainty-based outcome (DEC-003)")
con.execute(f"""
CREATE TABLE mcoh AS
SELECT w.stay_id, w.subject_id, w.intime, w.t24,
       fs.age_at_icu AS age, CASE WHEN fs.gender='M' THEN 1 ELSE 0 END AS sex_male,
       fs.dod,
       w.intime + INTERVAL 24 HOUR AS landmark,
       w.intime + INTERVAL 24 HOUR + INTERVAL 7 DAY AS window_end
FROM '{STG1}/win24.parquet' w
JOIN '{STG1}/first_stay.parquet' fs USING (stay_id)""")

con.execute("""
CREATE TABLE mclass AS
SELECT *,
  CASE dod IS NULL WHEN true THEN 'censored_no_death' ELSE
  CASE
    WHEN CAST(dod AS TIMESTAMP) + INTERVAL 1 DAY - INTERVAL 1 SECOND < landmark THEN 'definite_before_landmark'
    WHEN CAST(dod AS TIMESTAMP) > landmark AND CAST(dod AS TIMESTAMP) + INTERVAL 1 DAY - INTERVAL 1 SECOND <= window_end THEN 'definite_event'
    WHEN CAST(dod AS TIMESTAMP) > window_end THEN 'definite_nonevent_after_window'
    WHEN CAST(dod AS TIMESTAMP) <= landmark AND CAST(dod AS TIMESTAMP) + INTERVAL 1 DAY - INTERVAL 1 SECOND >= landmark THEN 'ambiguous_landmark'
    WHEN CAST(dod AS TIMESTAMP) <= window_end AND CAST(dod AS TIMESTAMP) + INTERVAL 1 DAY - INTERVAL 1 SECOND >= window_end THEN 'ambiguous_window_end'
    ELSE 'ambiguous_other' END
  END AS category
FROM mcoh""")

con.execute("""
CREATE TABLE mout AS
SELECT stay_id, subject_id, age, sex_male,
       CASE WHEN category='definite_event' THEN 1 ELSE 0 END AS y
FROM mclass
WHERE category NOT IN ('definite_before_landmark','ambiguous_landmark','ambiguous_window_end')""")
m_n, m_ev = con.execute("SELECT count(*), sum(y) FROM mout").fetchone()
print(f"[MIMIC] primary certainty-based cohort: N={m_n} events={m_ev} rate={100*m_ev/m_n:.3f}%")

ce_map = " ".join(f"WHEN itemid IN ({','.join(map(str, ids))}) THEN '{v}'"
                  for v, (ids, _, _) in VARS.items() if v in MIMIC_VITALS)
le_map = " ".join(f"WHEN itemid IN ({','.join(map(str, ids))}) THEN '{v}'"
                  for v, (ids, _, _) in VARS.items() if v not in MIMIC_VITALS)
rng_case = " ".join(f"WHEN '{v}' THEN (val BETWEEN {lo} AND {hi})" for v, (_, _, (lo, hi)) in VARS.items())

print("[MIMIC] events -> long (with hour index)")
con.execute(f"""
CREATE TABLE mlong AS
SELECT stay_id, var, t, val, FLOOR(t) AS hr FROM (
  SELECT stay_id, CASE {ce_map} END AS var, hrs_from_t0 AS t,
         CASE WHEN itemid=223761 THEN (valuenum-32.0)*5.0/9.0 ELSE valuenum END AS val
  FROM '{STG1}/ce_win48.parquet'
  WHERE hrs_from_t0 >= 0 AND hrs_from_t0 < 24 AND valuenum IS NOT NULL
  UNION ALL
  SELECT stay_id, CASE {le_map} END AS var, hrs_from_t0 AS t, valuenum AS val
  FROM '{STG1}/le_win48.parquet'
  WHERE hrs_from_t0 >= 0 AND hrs_from_t0 < 24 AND valuenum IS NOT NULL
) q
WHERE var IS NOT NULL AND (CASE var {rng_case} END)""")
mlong = con.execute("SELECT * FROM mlong").df()

# ============================================================== SICdb cohort/outcome (unchanged, no ambiguity)
print("[SICdb] cohort + outcome")
con.execute(f"""
CREATE TABLE scoh AS
SELECT CaseID, PatientID, ICUOffset, AgeOnAdmission AS age,
       CASE WHEN Sex=735 THEN 1 ELSE 0 END AS sex_male, OffsetOfDeath
FROM (
  SELECT CaseID, PatientID, COALESCE(ICUOffset,0) AS ICUOffset, AgeOnAdmission, Sex,
         OffsetOfDeath, TimeOfStay - COALESCE(ICUOffset,0) AS los,
         row_number() OVER (PARTITION BY PatientID
             ORDER BY COALESCE(OffsetAfterFirstAdmission,0), CaseID) rn
  FROM read_csv_auto('{SIC}/cases.csv.gz')
  WHERE AgeOnAdmission >= 18
) WHERE rn = 1 AND los/3600.0 >= 24""")
con.execute("""
CREATE TABLE sout AS
SELECT CaseID, age, sex_male,
       CASE WHEN OffsetOfDeath IS NOT NULL
             AND OffsetOfDeath >  ICUOffset+86400
             AND OffsetOfDeath <= ICUOffset+86400+7*86400 THEN 1 ELSE 0 END AS y
FROM scoh
WHERE OffsetOfDeath IS NULL OR OffsetOfDeath > ICUOffset+86400""")
s_n, s_ev = con.execute("SELECT count(*), sum(y) FROM sout").fetchone()
print(f"[SICdb] cohort: N={s_n} events={s_ev} rate={100*s_ev/s_n:.3f}%")

sv_map = " ".join(f"WHEN DataID IN ({','.join(map(str, ids))}) THEN '{v}'"
                  for v, (_, ids, _) in VARS.items() if v in MIMIC_VITALS)
sl_map = " ".join(f"WHEN LaboratoryID IN ({','.join(map(str, ids))}) THEN '{v}'"
                  for v, (_, ids, _) in VARS.items() if v not in MIMIC_VITALS)

print("[SICdb] events -> long (with hour index)")
con.execute(f"""
CREATE TABLE slong AS
SELECT CaseID, var, t, val, FLOOR(t) AS hr FROM (
  SELECT f.CaseID, CASE {sv_map} END AS var, f.t_sec/3600.0 AS t, f.Val AS val
  FROM '{OUT1}/tables/sicdb_float_h_window.parquet' f
  WHERE f.t_sec >= 0 AND f.t_sec < 86400 AND f.Val IS NOT NULL
  UNION ALL
  SELECT l.CaseID, CASE {sl_map} END AS var,
         (l.Offset - k.ICUOffset)/3600.0 AS t,
         CASE WHEN l.LaboratoryID = 355 THEN l.LaboratoryValue/{UREA_TO_BUN}
              ELSE l.LaboratoryValue END AS val
  FROM read_csv_auto('{SIC}/laboratory.csv.gz') l
  JOIN scoh k USING (CaseID)
  WHERE l.Offset >= k.ICUOffset AND l.Offset < k.ICUOffset + 86400
    AND l.LaboratoryValue IS NOT NULL
) q
WHERE var IS NOT NULL AND (CASE var {rng_case} END)""")
slong = con.execute("SELECT * FROM slong").df()

# ============================================================== shared aggregation helpers
def agg_A(long_df, key):
    """Representation A: per (key,var) first/last/min/max/median + observed_hour_density."""
    g = long_df.sort_values("t").groupby([key, "var"])
    first = g["val"].first().rename("first")
    last = g["val"].last().rename("last")
    mn = g["val"].min().rename("min")
    mx = g["val"].max().rename("max")
    med = g["val"].median().rename("median")
    dens = (long_df.groupby([key, "var"])["hr"].nunique() / 24.0).rename("density")
    w = pd.concat([first, last, mn, mx, med, dens], axis=1).reset_index()
    wide = w.pivot(index=key, columns="var")
    wide.columns = [f"{v}__{stat}" for stat, v in wide.columns]
    return wide


def agg_B(long_df, key):
    """Representation B: per (key,var,bin) median + bin_density, bin=0..3 (6h each)."""
    d = long_df.copy()
    d["bin"] = (d["hr"] // 6).astype(int).clip(0, 3)
    med = d.groupby([key, "var", "bin"])["val"].median().rename("median")
    dens = (d.groupby([key, "var", "bin"])["hr"].nunique() / 6.0).rename("bin_density")
    w = pd.concat([med, dens], axis=1).reset_index()
    wide = w.pivot(index=key, columns=["var", "bin"])
    wide.columns = [f"{v}__b{b}__{stat}" for stat, v, b in wide.columns]
    return wide


print("[extract] Representation A (both DBs)")
mA = agg_A(mlong, "stay_id")
sA = agg_A(slong, "CaseID")
print("[extract] Representation B (both DBs)")
mB = agg_B(mlong, "stay_id")
sB = agg_B(slong, "CaseID")

mout_df = con.execute("SELECT * FROM mout").df().set_index("stay_id")
sout_df = con.execute("SELECT * FROM sout").df().set_index("CaseID")

mA_full = mout_df.join(mA, how="left")
sA_full = sout_df.join(sA, how="left")
mB_full = mout_df.join(mB, how="left")
sB_full = sout_df.join(sB, how="left")

# ---- fill structural NaN (variable never measured) and derive missingness/masks ----
A_stats = ["first", "last", "min", "max", "median"]
for d in (mA_full, sA_full):
    for v in VARS:
        for s in A_stats:
            c = f"{v}__{s}"
            if c not in d.columns:
                d[c] = np.nan
        dc = f"{v}__density"
        if dc not in d.columns:
            d[dc] = 0.0
        else:
            d[dc] = d[dc].fillna(0.0)
        d[f"{v}__missing"] = (d[dc] == 0.0).astype(int)

for d in (mB_full, sB_full):
    for v in VARS:
        for b in range(4):
            mc = f"{v}__b{b}__median"
            if mc not in d.columns:
                d[mc] = np.nan
            dc = f"{v}__b{b}__bin_density"
            if dc not in d.columns:
                d[dc] = 0.0
            else:
                d[dc] = d[dc].fillna(0.0)
            d[f"{v}__b{b}__mask"] = (d[dc] > 0.0).astype(int)

os.makedirs(STG2, exist_ok=True)
mA_full.reset_index().to_parquet(f"{STG2}/repA_mimic.parquet")
sA_full.reset_index().to_parquet(f"{STG2}/repA_sicdb.parquet")
mB_full.reset_index().to_parquet(f"{STG2}/repB_mimic.parquet")
sB_full.reset_index().to_parquet(f"{STG2}/repB_sicdb.parquet")

meta = {
    "mimic_primary_cohort": {"n": int(m_n), "events": int(m_ev), "rate_pct": round(100 * m_ev / m_n, 4)},
    "sicdb_cohort": {"n": int(s_n), "events": int(s_ev), "rate_pct": round(100 * s_ev / s_n, 4)},
    "n_core_variables": len(VARS),
    "repA_value_stats": A_stats,
    "repB_bins": ["0-6h", "6-12h", "12-18h", "18-24h"],
    "repB_per_bin_stats": ["median", "bin_density", "mask"],
    "density_definition": "DEC-002: distinct observed hours / 24 (RepA) or /6 per bin (RepB); "
                           "duplicate measurements within an hour count once",
    "no_raw_count_features": True,
}
with open(f"{OUT2}/qc/pilot_extract_meta_2a.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(json.dumps(meta, indent=2))
print("DONE")
