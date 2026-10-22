# Decision consequences of evaluating synthetic clinical-data suitability across stochastic realizations and hospitals

## Purpose

A methodological evaluation of synthetic clinical-data suitability for a fixed ICU mortality-prediction task across stochastic synthetic-data realizations and destination hospitals. Error, decisiveness, false qualification and interval coverage are interpreted jointly; no strategy is optimal or universally superior.

## Study architecture

MIMIC-IV v3.1 supplies source training and held-out internal evaluation; SICdb v1.0.8 supplies external evaluation; eICU-CRD v2.0 supplies multicenter external evaluation and eligible hospital populations. GaussianCopula, CTGAN and TabDDPM each have 15 generation attempts. Utility-estimable realizations are **15/15, 15/15 and 14/15**, respectively. One TabDDPM attempt has single-class collapse and remains in the reliability denominator. Learners are L2 logistic regression and XGBoost, with fixed specifications.

Time zero is ICU admission; prediction is at 24 hours using the [0,24 h) measurement window. The mortality endpoints are replication-compatible, not strictly harmonized. H100, H200 and H300 contain 127, 85 and 66 hospitals; H200 is primary. Different thresholds define different target hospital populations.

## Future-use targets

External utility loss (EUL) is destination AUROC of the real-trained model minus destination AUROC of the corresponding synthetic-trained model. Primary **Target B** is latent EUL for a new utility-estimable realization at a future hospital exchangeable with the H200-eligible population. Secondary **Target A** averages over the realization distribution at a future eligible hospital. Future finite-sample AUROC measurement error is excluded from these latent targets.

## Evaluation strategies

- **M1:** all estimable realizations; crossed realization, hospital and interaction effects; full within-hospital sampling covariance.
- **M2:** the same crossed model and data, with diagonal sampling covariance.
- **M3:** realizations aggregated before hospital-only inference; Target-B output is naive extrapolation.
- **M4:** one randomly selected realization across hospitals; Target-B output is naive extrapolation.

P95 is primary: qualify if P(EUL≤δ)≥0.95, reject if ≤0.05, otherwise indeterminate. Tolerances are 0.025, 0.05, 0.10 and 0.20; they are methodological stress tolerances, not clinical standards. Confidence sensitivity at 0.80, 0.90, 0.95, 0.975 and 0.99 is separate from model architecture. C95 uses central-95% interval endpoints. Main quantities are DER, DY, CDY, FQ-all, macro-SER, macro FQ-def and coverage.

## What this repository contains

Current target/decision and known-truth simulation code in `decision_evaluation/`; reproducibility configs; approved aggregate manuscript source data and final figures; a renderer for Figures 1–5 and the supplementary confidence trajectories; fabricated fixtures and tests; and supporting source-pipeline interfaces. See [workflow](docs/ANALYSIS_WORKFLOW.md) and [scope](docs/PUBLIC_RELEASE_SCOPE.md).

## What this repository does not contain

Row-level MIMIC-IV, SICdb or eICU records; evaluated synthetic patient rows derived from restricted sources; patient predictions; trained clinical model objects or weights; private bootstrap indices; source hospital identifiers or restricted hospital-indexed derivatives; credentials; private project records; manuscript drafts; or superseded analytical outputs.

## Reproduction boundary

**A — Public fabricated tests:** run the software tests and portable fabricated example without clinical access.

**B — Aggregate manuscript displays:** render main displays from `manuscript_source_data/` without clinical records or refitting:

```sh
python -m pip install -r requirements-display.txt
python reproduction/verify_public_release.py
python reproduction/reproduce_main_displays.py --output reproduced_displays
python reproduction/verify_public_release.py --rendered reproduced_displays
```

**C — Authorized source reconstruction:** clinical training/evaluation and empirical-calibrated simulation execution require separately authorized database access plus the specified restricted input interfaces and historical dependencies. Public aggregate display reproduction does not establish complete raw-to-result clinical replay. No clinical analysis, production simulation or scientific model was rerun to prepare this release.

## Public release

Current release: **v1.3.0-manuscript-aligned**. Stable archive: [Zenodo concept DOI 10.5281/zenodo.22822481](https://doi.org/10.5281/zenodo.22822481). The exact version DOI is recorded with the immutable GitHub release after archive publication.

## Citation

See [CITATION.cff](CITATION.cff).

## License and data governance

Original software is MIT licensed. Released aggregate results/documentation retain CC BY 4.0. Third-party software and clinical databases retain their own licenses and access requirements; a software license does not grant clinical-data rights. See [data access](docs/DATA_ACCESS.md).
