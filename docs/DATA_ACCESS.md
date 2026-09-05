# Data access

This is a code-only repository. It does not redistribute MIMIC-IV, SICdb, eICU-CRD, synthetic datasets, participant-level predictions, or fitted models.

Researchers must obtain each source database independently and comply with its current credentialing, data-use, and citation requirements. Nothing in this repository bypasses those controls. The 45 synthetic row-level datasets are planned for separate PhysioNet deposition under applicable access conditions; the DOI is pending.

The source scripts accept local locations through environment variables:

```bash
export MIMIC_HOSP_ROOT=/path/to/mimic/hosp
export MIMIC_ICU_ROOT=/path/to/mimic/icu
export SICDB_ROOT=/path/to/sicdb
export EICU_MASTER_PARQUET=/path/to/authorized/eicu/master_icu_stay.parquet
export ICU_PROJECT_ROOT=/path/to/this/repository
export ICU_WORK_ROOT=/path/to/derived/workspace
```

`EICU_MASTER_PARQUET` refers to the study’s canonical, locally constructed nested-event asset. Its row-level builder is not distributed here because it belongs to separately governed source-data infrastructure. This is a documented reproduction dependency, not an included dataset.
