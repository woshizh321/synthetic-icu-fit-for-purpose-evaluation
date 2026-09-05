"""PREFLIGHT-1 / MIMIC-IV v3.1 : cohort cascade + time-zero audit (read-only).

Age rule (MIMIC-IV date-shift aware):
    age_at_icu_admission = patients.anchor_age + (year(icustays.intime) - patients.anchor_year)
anchor_year is the shifted year in which the patient was anchor_age years old, and all
event times for that subject are shifted onto the same axis, so the year difference is
valid within-subject. anchor_year_group is retained for era reporting only.
"""
import json
import os
import random

import duckdb
import numpy as np

random.seed(42)
np.random.seed(42)

HOSP = os.environ["MIMIC_HOSP_ROOT"]
ICU = os.environ["MIMIC_ICU_ROOT"]
OUT = os.path.join(os.environ.get("ICU_WORK_ROOT", "outputs"), "preflight1")
STG = f"{OUT}/staging"
os.makedirs(STG, exist_ok=True)

con = duckdb.connect(f"{STG}/preflight.duckdb")
con.execute("PRAGMA memory_limit='12GB'; PRAGMA threads=10;")

con.execute(f"""
CREATE OR REPLACE TABLE patients AS SELECT * FROM read_csv_auto('{HOSP}/patients.csv.gz');
CREATE OR REPLACE TABLE admissions AS SELECT * FROM read_csv_auto('{HOSP}/admissions.csv.gz');
CREATE OR REPLACE TABLE icustays AS SELECT * FROM read_csv_auto('{ICU}/icustays.csv.gz');
CREATE OR REPLACE TABLE d_items AS SELECT * FROM read_csv_auto('{ICU}/d_items.csv.gz');
CREATE OR REPLACE TABLE d_labitems AS SELECT * FROM read_csv_auto('{HOSP}/d_labitems.csv.gz');
""")

con.execute("""
CREATE OR REPLACE TABLE stays AS
SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.first_careunit, i.last_careunit,
       i.los AS los_days_native,
       date_diff('second', i.intime, i.outtime)/3600.0 AS los_hours,
       p.anchor_age, p.anchor_year, p.anchor_year_group, p.gender, p.dod,
       p.anchor_age + (extract(year FROM i.intime) - p.anchor_year) AS age_at_icu,
       a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.race, a.admission_type
FROM icustays i
JOIN patients p USING (subject_id)
LEFT JOIN admissions a USING (subject_id, hadm_id);
""")

q = con.execute
n_stays = q("SELECT count(*) FROM stays").fetchone()[0]
n_pat = q("SELECT count(DISTINCT subject_id) FROM stays").fetchone()[0]

n_adult_stays = q("SELECT count(*) FROM stays WHERE age_at_icu >= 18").fetchone()[0]
n_adult_pat = q("SELECT count(DISTINCT subject_id) FROM stays WHERE age_at_icu >= 18").fetchone()[0]
n_under18 = q("SELECT count(*) FROM stays WHERE age_at_icu < 18").fetchone()[0]
n_age_null = q("SELECT count(*) FROM stays WHERE age_at_icu IS NULL").fetchone()[0]

# First eligible ICU stay per patient: earliest intime, tie-break by smallest stay_id.
con.execute("""
CREATE OR REPLACE TABLE first_stay AS
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime ASC, stay_id ASC) AS rn
  FROM stays WHERE age_at_icu >= 18
) WHERE rn = 1;
""")
n_first = q("SELECT count(*) FROM first_stay").fetchone()[0]
n_ties = q("""SELECT count(*) FROM (SELECT subject_id, intime FROM stays WHERE age_at_icu>=18
              GROUP BY 1,2 HAVING count(*)>1)""").fetchone()[0]

con.execute("CREATE OR REPLACE TABLE cohort24 AS SELECT * FROM first_stay WHERE los_hours >= 24;")
con.execute("CREATE OR REPLACE TABLE cohort48 AS SELECT * FROM first_stay WHERE los_hours >= 48;")
n24 = q("SELECT count(*) FROM cohort24").fetchone()[0]
n48 = q("SELECT count(*) FROM cohort48").fetchone()[0]

# ---- time zero audit ----
tz = q("""
SELECT
 sum(CASE WHEN intime IS NULL THEN 1 ELSE 0 END) AS n_null_intime,
 sum(CASE WHEN outtime IS NULL THEN 1 ELSE 0 END) AS n_null_outtime,
 sum(CASE WHEN outtime < intime THEN 1 ELSE 0 END) AS n_outtime_before_intime,
 sum(CASE WHEN los_hours <= 0 THEN 1 ELSE 0 END) AS n_los_nonpositive,
 sum(CASE WHEN los_hours > 8760 THEN 1 ELSE 0 END) AS n_los_gt_1yr,
 sum(CASE WHEN hadm_id IS NULL THEN 1 ELSE 0 END) AS n_null_hadm,
 min(los_hours), quantile_cont(los_hours,0.05), quantile_cont(los_hours,0.5),
 quantile_cont(los_hours,0.95), max(los_hours), count(*)
FROM stays;
""").fetchone()
tz_keys = ["n_null_intime","n_null_outtime","n_outtime_before_intime","n_los_nonpositive",
           "n_los_gt_1yr","n_null_hadm","los_h_min","los_h_p5","los_h_p50","los_h_p95",
           "los_h_max","n_total"]
tz_d = dict(zip(tz_keys, [float(x) if x is not None else None for x in tz]))

multi = q("""SELECT count(*) FROM (SELECT hadm_id FROM icustays WHERE hadm_id IS NOT NULL
             GROUP BY 1 HAVING count(*)>1)""").fetchone()[0]
n_hadm = q("SELECT count(DISTINCT hadm_id) FROM icustays").fetchone()[0]
multi_pat = q("""SELECT count(*) FROM (SELECT subject_id FROM icustays GROUP BY 1 HAVING count(*)>1)""").fetchone()[0]

# staging tables used by later scripts (48h window = superset of 24h)
con.execute("""
CREATE OR REPLACE TABLE win AS
SELECT stay_id, subject_id, hadm_id, intime,
       intime + INTERVAL 24 HOUR AS t24,
       intime + INTERVAL 48 HOUR AS t48,
       los_hours, outtime
FROM first_stay WHERE los_hours >= 24;
""")
con.execute(f"COPY win TO '{STG}/win24.parquet' (FORMAT PARQUET);")
con.execute(f"COPY (SELECT * FROM first_stay) TO '{STG}/first_stay.parquet' (FORMAT PARQUET);")

summary = dict(n_icustays_all=n_stays, n_patients_all=n_pat, n_hadm_with_icu=n_hadm,
               n_adult_stays=n_adult_stays, n_adult_patients=n_adult_pat,
               n_stays_age_lt18=n_under18, n_stays_age_null=n_age_null,
               n_first_stay_adult=n_first, n_intime_ties=n_ties,
               n_first_stay_los_ge24h=n24, n_first_stay_los_ge48h=n48,
               n_hadm_with_multiple_icustays=multi, n_patients_with_multiple_icustays=multi_pat,
               time_zero_audit=tz_d)
with open(f"{OUT}/qc/mimic_cohort_timezero.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

rows = [
 ("all_icu_stays", n_stays, None, "icustays.csv.gz, all rows"),
 ("unique_patients", n_pat, None, "distinct subject_id among ICU stays"),
 ("adult_icu_stays_age_ge18", n_adult_stays, 100.0*n_adult_stays/n_stays,
  "age = anchor_age + (year(intime) - anchor_year); n<18 = %d, n_null = %d" % (n_under18, n_age_null)),
 ("adult_unique_patients", n_adult_pat, None, "distinct subject_id among adult stays"),
 ("first_icu_stay_per_adult_patient", n_first, 100.0*n_first/n_adult_stays,
  "earliest intime per subject_id, tie-break min(stay_id); intime ties=%d" % n_ties),
 ("first_stay_los_ge_24h", n24, 100.0*n24/n_first, "los_hours = (outtime-intime)"),
 ("first_stay_los_ge_48h", n48, 100.0*n48/n24, "pct relative to >=24h cohort"),
]
import csv
with open(f"{OUT}/tables/mimic_cohort_cascade.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step","n_icustays_or_patients","pct_retained_from_previous","notes"])
    for r in rows:
        w.writerow([r[0], r[1], "" if r[2] is None else round(r[2],2), r[3]])

print(json.dumps(summary, indent=2, default=str))
con.close()
