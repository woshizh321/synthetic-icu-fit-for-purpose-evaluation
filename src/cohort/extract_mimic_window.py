"""PREFLIGHT-1 / MIMIC-IV: ONE streaming pass over chartevents + labevents.

Extracts only rows belonging to the adult-first-ICU-stay >=24h cohort and only
those with charttime inside [intime, intime+48h]. All itemids are retained so
that itemid selection can be done afterwards on the small staged parquet
without re-scanning the raw gz files. Read-only against the raw database.
"""
import os
import random
import time

import duckdb
import numpy as np

random.seed(42)
np.random.seed(42)

HOSP = os.environ["MIMIC_HOSP_ROOT"]
ICU = os.environ["MIMIC_ICU_ROOT"]
STG = os.path.join(os.environ.get("ICU_WORK_ROOT", "outputs"), "preflight1", "staging")

con = duckdb.connect(f"{STG}/preflight.duckdb")
con.execute("PRAGMA memory_limit='12GB'; PRAGMA threads=10;")
con.execute("PRAGMA preserve_insertion_order=false;")
con.execute(f"PRAGMA temp_directory='{STG}/duckdb_tmp';")

CE_TYPES = {"subject_id":"BIGINT","hadm_id":"BIGINT","stay_id":"BIGINT","caregiver_id":"BIGINT",
            "charttime":"TIMESTAMP","storetime":"TIMESTAMP","itemid":"BIGINT","value":"VARCHAR",
            "valuenum":"DOUBLE","valueuom":"VARCHAR","warning":"BIGINT"}
LE_TYPES = {"labevent_id":"BIGINT","subject_id":"BIGINT","hadm_id":"BIGINT","specimen_id":"BIGINT",
            "itemid":"BIGINT","order_provider_id":"VARCHAR","charttime":"TIMESTAMP",
            "storetime":"TIMESTAMP","value":"VARCHAR","valuenum":"DOUBLE","valueuom":"VARCHAR",
            "ref_range_lower":"DOUBLE","ref_range_upper":"DOUBLE","flag":"VARCHAR",
            "priority":"VARCHAR","comments":"VARCHAR"}

t = time.time()
con.execute(f"""
COPY (
  SELECT c.stay_id, c.subject_id, c.itemid, c.charttime, c.storetime,
         c.value, c.valuenum, c.valueuom, c.warning,
         date_diff('second', w.intime, c.charttime)/3600.0 AS hrs_from_t0
  FROM read_csv('{ICU}/chartevents.csv.gz', columns={CE_TYPES}, header=true, compression='gzip',
                quote='"', escape='"', strict_mode=false) c
  JOIN win w ON c.stay_id = w.stay_id
  WHERE c.charttime >= w.intime AND c.charttime <= w.t48
) TO '{STG}/ce_win48.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
""")
print(f"chartevents pass done in {time.time()-t:.0f}s", flush=True)

t = time.time()
con.execute(f"""
COPY (
  SELECT w.stay_id, l.subject_id, l.hadm_id, l.itemid, l.charttime, l.storetime,
         l.value, l.valuenum, l.valueuom, l.flag,
         date_diff('second', w.intime, l.charttime)/3600.0 AS hrs_from_t0
  FROM read_csv('{HOSP}/labevents.csv.gz', columns={LE_TYPES}, header=true, compression='gzip',
                quote='"', escape='"', strict_mode=false) l
  JOIN win w ON l.subject_id = w.subject_id
  WHERE l.charttime >= w.intime AND l.charttime <= w.t48
) TO '{STG}/le_win48.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
""")
print(f"labevents pass done in {time.time()-t:.0f}s", flush=True)

for nm, path in [("ce", "ce_win48.parquet"), ("le", "le_win48.parquet")]:
    n = con.execute(f"SELECT count(*) FROM read_parquet('{STG}/{path}')").fetchone()[0]
    print(nm, "rows in 48h window:", n, flush=True)
con.close()
