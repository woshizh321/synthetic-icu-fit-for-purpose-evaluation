# GitHub release audit

## Repository

- Owner: `woshizh321`
- Repository: `synthetic-icu-fit-for-purpose-evaluation`
- URL: `https://github.com/woshizh321/synthetic-icu-fit-for-purpose-evaluation`
- Visibility: `PRIVATE`
- Branch: `main`
- Reviewed core-code commit: `22b6ff89e57330c4fb3a989238347adee8263e66`

## Included

- `src/cohort` and `src/harmonization`: source cohort, first-24-hour representation, 150-field training contract, and corrected eICU-CRD extraction semantics.
- `src/generators`: final SDV GaussianCopula/CTGAN and non-EMA TabDDPM production code.
- `src/models`: exact logistic-regression and XGBoost learner kernels.
- `src/estimands` and `src/bootstrap`: IUL/EUL/ITL, proper-score skill transformations, calibration, and 2,000-replicate hierarchical bootstrap.
- `src/multicenter`: outcome-independent D_W and full within-hospital seed×seed covariance crossed-effects REML.
- `src/fidelity`, `src/measurement_process`, and `src/privacy`: separate fidelity, structure, measurement-process, and empirical-attack kernels.
- `src/reporting`: editable SVG/PDF and 450-dpi PNG Figure 1 generator.
- `configs`, `docs`, requirements files, `.gitignore`, and fabricated-data tests.

## Excluded

Confirmed absent from the Git index:

- MIMIC-IV, SICdb, and eICU-CRD row-level data.
- All 45 synthetic row-level datasets.
- Participant-level predictions and probabilities.
- Model checkpoints, fitted transformers, and serialized models.
- Bootstrap participant indices and row-level hospital outputs.
- Nearest-neighbor and membership-inference record-level outputs.
- Credentials, secrets, authentication files, salts, and `.env` files.
- Manuscript drafts, PI adjudications, internal review reports, archives, and raw logs.

Clinical identifier names such as `subject_id` and `patientunitstayid` occur only as schema/SQL field names; no identifier values are embedded.

## Checks

- Local absolute-path scan (`/Users/hezhu`, `/mnt`, drive-letter paths): PASS.
- Credential-pattern scan (`password`, `token`, `api_key`, private-key markers, GitHub token prefix): PASS.
- Prohibited data/model extension scan across tracked files: PASS.
- Large-file scan (>1 MiB): PASS; no matches.
- `git diff --cached --check`: PASS.
- Python AST parse: PASS, 24 files.
- Fabricated-data tests: PASS, 3 tests.
- Real clinical analysis, generation, model fitting, and result recomputation: NOT RUN, as required.

## Reproducibility and remaining gaps

- Authoritative scripts, configs, environment notes, estimand definitions, and execution order are included.
- External access to the three source databases is required.
- The canonical nested eICU-CRD asset builder belongs to separate data infrastructure and is not included.
- Historical fitted objects and participant-level prediction artifacts are not distributed.
- Transitive historical package versions were not fully recoverable; known exact versions are separated by stage.
- `LICENSE_REQUIRES_PI_DECISION`.
- `AUTHOR_METADATA_PENDING_PI`; `CITATION.cff` deferred.
- PhysioNet DOI pending.

## Pre-push verdict

`GITHUB_RELEASE_SAFETY_AUDIT_PASS`
