# Analysis workflow

The order below reflects the locked scientific dependency graph. Scripts are intentionally close to production sources rather than refactored into a new framework.

1. Build the adult first-ICU-stay MIMIC-IV cohort and stage events through 48 h (`src/cohort`).
2. Derive the static first-24-hour representation for MIMIC-IV and SICdb, then construct the 150-field MIMIC-IV training contract (`src/harmonization`).
3. Build the eICU-CRD contract from the authorized canonical nested-event asset. Observed-hour density uses `FLOOR(offset/60)` for `0 <= offset < 1440` (`src/harmonization/build_eicu_contract.py`).
4. Generate 15 realizations per generator with seeds 42–56 (`src/generators`). The TabDDPM outcome is categorical; production sampling is non-EMA. Do not repair collapse, substitute seeds, or condition sampling on the outcome.
5. Record all 45 attempts for reliability. Train downstream models only for utility-estimable realizations, using `src/models/frozen_learners.py`.
6. Produce identically ordered real/synthetic prediction matrices for MIMIC internal, SICdb external, and eICU-CRD external samples. Prediction artifacts are intentionally excluded from GitHub.
7. Compute AUROC, AP skill, Brier skill, log-loss skill, and calibration (`src/estimands`).
8. Run generator-level hierarchical bootstrap (`src/bootstrap/generator_hierarchical_bootstrap.py`).
9. Lock outcome-independent `D_W`, create paired patient-bootstrap full covariance blocks, and fit the crossed-effects REML (`src/multicenter`).
10. Compute fidelity, structural validity, measurement-process, and empirical privacy diagnostics as separate dimensions (`src/fidelity`, `src/measurement_process`, `src/privacy`).
11. Generate Figure 1 using `src/reporting/generate_figure1.py`. Other publication figures require reviewed aggregate result inputs that are not redistributed in this code-only release.

The U2 scripts expect the historical production artifact layout under `outputs/postreview_upgrade/...`. This layout is retained to preserve traceability. Users may recreate it after obtaining source data, or adapt only path plumbing while preserving ordered row identities, estimands, resampling, and covariance definitions.
