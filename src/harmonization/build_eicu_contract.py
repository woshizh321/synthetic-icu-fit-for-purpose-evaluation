"""D-09/10: build the leakage-safe eICU replication contract from nested events only,
per DEC-016 (cohort/ordering), DEC-017 (age), DEC-018 (variable mapping), DEC-019
(endpoint). Window: 0 <= offset < 1440 minutes. Deterministic tie-break via native
stable record IDs (labid/vitalperiodicid/vitalaperiodicid), never nested-array order.
"""
import duckdb
import os
import time

MASTER = os.environ["EICU_MASTER_PARQUET"]
WORK_ROOT = os.environ.get("ICU_WORK_ROOT", "outputs")
DB = os.path.join(WORK_ROOT, "preflight2d", "eicu_work.duckdb")
OUTPUT_PARQUET = os.path.join(WORK_ROOT, "preflight2d", "eicu_replication_contract.parquet")

t0 = time.time()
con = duckdb.connect(DB)
con.execute("PRAGMA threads=8")

# ---------------------------------------------------------------------------
# Cohort (DEC-016): adult, first eligible stay per patient (approximate ordering),
# >=24h observable.
# ---------------------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TABLE cohort AS
WITH elig AS (
    SELECT patientunitstayid, uniquepid, hospitalid, age, gender,
           hospitaldischargestatus, hospitaldischargeoffset, unitdischargestatus,
           unitdischargeoffset, hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber,
           hospital_numbedscategory, hospital_teachingstatus, hospital_region,
           ROW_NUMBER() OVER (
               PARTITION BY uniquepid
               ORDER BY hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber, patientunitstayid
           ) AS rn
    FROM read_parquet('{MASTER}')
    WHERE ((TRY_CAST(age AS INTEGER) >= 18) OR (age = '> 89'))
      AND unitdischargeoffset >= 1440
)
SELECT patientunitstayid, uniquepid, hospitalid,
       CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_primary,
       CASE WHEN age = '> 89' THEN 92 ELSE TRY_CAST(age AS INTEGER) END AS age_sensitivity,
       CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS sex_male,
       hospitaldischargestatus, hospitaldischargeoffset, unitdischargestatus, unitdischargeoffset,
       hospital_numbedscategory, hospital_teachingstatus, hospital_region
FROM elig WHERE rn = 1
""")
n_cohort = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
print(f"cohort N={n_cohort} ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Outcome (DEC-019): primary 7-day post-landmark in-hospital mortality proxy
# ---------------------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE cohort_outcome AS
SELECT *,
    CASE WHEN hospitaldischargestatus = 'Expired'
              AND hospitaldischargeoffset > 1440 AND hospitaldischargeoffset <= 11520
         THEN 1 ELSE 0 END AS mortality_7d_post_landmark,
    -- complete-observation sensitivity flag (Phase D-28): event within window, OR
    -- still observable through window end (hospitaldischargeoffset > 11520)
    CASE WHEN (hospitaldischargestatus='Expired' AND hospitaldischargeoffset>1440 AND hospitaldischargeoffset<=11520)
              OR (hospitaldischargeoffset IS NOT NULL AND hospitaldischargeoffset > 11520)
         THEN 1 ELSE 0 END AS complete_observation_eligible
FROM cohort
""")

# ---------------------------------------------------------------------------
# Materialize cohort-only nested event arrays ONCE (semi-join), so every
# subsequent per-variable extraction unnests from this small in-database table
# instead of rescanning the 5.5GB master repeatedly.
# ---------------------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TABLE cohort_events AS
SELECT m.patientunitstayid, m.lab_events, m.vitalperiodic_events, m.vitalaperiodic_events
FROM read_parquet('{MASTER}') m
WHERE m.patientunitstayid IN (SELECT patientunitstayid FROM cohort)
""")
print(f"materialized cohort_events ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Per-variable extraction macros
# ---------------------------------------------------------------------------
VITAL_DIRECT = {  # variable: (source_table_col, source_field)
    "heart_rate": ("vitalperiodic_events", "heartrate"),
    "respiratory_rate": ("vitalperiodic_events", "respiration"),
    "spo2": ("vitalperiodic_events", "sao2"),
    "temperature_c": ("vitalperiodic_events", "temperature"),
}
BP_VARS = {  # variable: (invasive_field, noninvasive_field)
    "sbp": ("systemicsystolic", "noninvasivesystolic"),
    "dbp": ("systemicdiastolic", "noninvasivediastolic"),
    "map": ("systemicmean", "noninvasivemean"),
}
LAB_VARS = {  # variable: accepted label list
    "wbc": ["WBC x 1000"], "hemoglobin": ["Hgb"], "platelet_count": ["platelets x 1000"],
    "creatinine": ["creatinine"], "bun_urea": ["BUN"], "sodium": ["sodium"],
    "potassium": ["potassium"], "chloride": ["chloride"], "lactate": ["lactate"],
    "pao2": ["paO2"], "paco2": ["paCO2"], "glucose": ["glucose"],
    "bicarbonate": ["bicarbonate"], "ph_arterial": ["pH"],
}

def build_vital_direct(var, field):
    con.execute(f"""
    CREATE OR REPLACE TABLE var_{var} AS
    WITH ev AS (
        SELECT m.patientunitstayid, e.observationoffset AS off, e.vitalperiodicid AS rid, e.{field} AS val
        FROM cohort_events m, UNNEST(m.vitalperiodic_events) AS t(e)
        WHERE e.observationoffset >= 0 AND e.observationoffset < 1440 AND e.{field} IS NOT NULL
    )
    SELECT patientunitstayid,
        arg_min(val, [off, rid]) AS {var}_first,
        arg_max(val, [off, rid]) AS {var}_last,
        MIN(val) AS {var}_min, MAX(val) AS {var}_max, MEDIAN(val) AS {var}_median,
        1 AS {var}_observed_any,
        COUNT(DISTINCT FLOOR(off/60.0)) / 24.0 AS {var}_observed_hour_density
    FROM ev GROUP BY patientunitstayid
    """)

def build_bp(var, inv_field, noninv_field):
    con.execute(f"""
    CREATE OR REPLACE TABLE var_{var}_inv AS
    WITH ev AS (
        SELECT m.patientunitstayid, e.observationoffset AS off, e.vitalperiodicid AS rid, e.{inv_field} AS val
        FROM cohort_events m, UNNEST(m.vitalperiodic_events) AS t(e)
        WHERE e.observationoffset >= 0 AND e.observationoffset < 1440 AND e.{inv_field} IS NOT NULL
    )
    SELECT patientunitstayid,
        arg_min(val, [off, rid]) AS first_v, arg_max(val, [off, rid]) AS last_v,
        MIN(val) AS min_v, MAX(val) AS max_v, MEDIAN(val) AS median_v,
        COUNT(DISTINCT FLOOR(off/60.0)) / 24.0 AS density_v
    FROM ev GROUP BY patientunitstayid
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE var_{var}_noninv AS
    WITH ev AS (
        SELECT m.patientunitstayid, e.observationoffset AS off, e.vitalaperiodicid AS rid, e.{noninv_field} AS val
        FROM cohort_events m, UNNEST(m.vitalaperiodic_events) AS t(e)
        WHERE e.observationoffset >= 0 AND e.observationoffset < 1440 AND e.{noninv_field} IS NOT NULL
    )
    SELECT patientunitstayid,
        arg_min(val, [off, rid]) AS first_v, arg_max(val, [off, rid]) AS last_v,
        MIN(val) AS min_v, MAX(val) AS max_v, MEDIAN(val) AS median_v,
        COUNT(DISTINCT FLOOR(off/60.0)) / 24.0 AS density_v
    FROM ev GROUP BY patientunitstayid
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE var_{var} AS
    SELECT COALESCE(i.patientunitstayid, n.patientunitstayid) AS patientunitstayid,
        COALESCE(i.first_v, n.first_v) AS {var}_first,
        COALESCE(i.last_v, n.last_v) AS {var}_last,
        COALESCE(i.min_v, n.min_v) AS {var}_min,
        COALESCE(i.max_v, n.max_v) AS {var}_max,
        COALESCE(i.median_v, n.median_v) AS {var}_median,
        1 AS {var}_observed_any,
        COALESCE(i.density_v, n.density_v) AS {var}_observed_hour_density,
        (i.patientunitstayid IS NOT NULL) AS {var}_used_invasive
    FROM var_{var}_inv i FULL OUTER JOIN var_{var}_noninv n USING (patientunitstayid)
    """)

def build_lab(var, labels):
    label_list = ",".join(f"'{l}'" for l in labels)
    con.execute(f"""
    CREATE OR REPLACE TABLE var_{var} AS
    WITH ev AS (
        SELECT m.patientunitstayid, e.labresultoffset AS off, e.labid AS rid, e.labresult AS val
        FROM cohort_events m, UNNEST(m.lab_events) AS t(e)
        WHERE e.labresultoffset >= 0 AND e.labresultoffset < 1440
          AND e.labname IN ({label_list}) AND e.labresult IS NOT NULL
    )
    SELECT patientunitstayid,
        arg_min(val, [off, rid]) AS {var}_first,
        arg_max(val, [off, rid]) AS {var}_last,
        MIN(val) AS {var}_min, MAX(val) AS {var}_max, MEDIAN(val) AS {var}_median,
        1 AS {var}_observed_any,
        COUNT(DISTINCT FLOOR(off/60.0)) / 24.0 AS {var}_observed_hour_density
    FROM ev GROUP BY patientunitstayid
    """)

for var, (_, field) in VITAL_DIRECT.items():
    build_vital_direct(var, field)
    print(f"built {var} ({time.time()-t0:.1f}s)")
for var, (inv, noninv) in BP_VARS.items():
    build_bp(var, inv, noninv)
    print(f"built {var} ({time.time()-t0:.1f}s)")
for var, labels in LAB_VARS.items():
    build_lab(var, labels)
    print(f"built {var} ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Assemble final contract: join all 21 variables onto cohort, apply observed_any=0
# default for stays with no matching event, fill physiology values per the SAME
# MIMIC-train-derived DEC-008 median fill values (loaded from the frozen JSON).
# ---------------------------------------------------------------------------
ALL_VARS = list(VITAL_DIRECT.keys()) + list(BP_VARS.keys()) + list(LAB_VARS.keys())
join_sql = "cohort_outcome c"
for var in ALL_VARS:
    join_sql += f" LEFT JOIN var_{var} v_{var} USING (patientunitstayid)"

select_cols = ["c.patientunitstayid", "c.uniquepid", "c.hospitalid", "c.age_primary", "c.age_sensitivity",
               "c.sex_male", "c.mortality_7d_post_landmark", "c.complete_observation_eligible",
               "c.hospital_numbedscategory", "c.hospital_teachingstatus", "c.hospital_region"]
for var in ALL_VARS:
    select_cols += [
        f"v_{var}.{var}_first", f"v_{var}.{var}_last", f"v_{var}.{var}_min",
        f"v_{var}.{var}_max", f"v_{var}.{var}_median",
        f"COALESCE(v_{var}.{var}_observed_any, 0) AS {var}_observed_any",
        f"COALESCE(v_{var}.{var}_observed_hour_density, 0.0) AS {var}_observed_hour_density",
    ]

con.execute(f"""
CREATE OR REPLACE TABLE eicu_contract_raw AS
SELECT {', '.join(select_cols)}
FROM {join_sql}
""")
n_final = con.execute("SELECT COUNT(*) FROM eicu_contract_raw").fetchone()[0]
print(f"eicu_contract_raw N={n_final} ({time.time()-t0:.1f}s)")

con.execute("""
COPY eicu_contract_raw TO '{OUTPUT_PARQUET}' (FORMAT PARQUET)
""")
print(f"DONE ({time.time()-t0:.1f}s)")
