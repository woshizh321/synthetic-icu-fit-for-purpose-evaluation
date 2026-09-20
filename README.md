# Stochasticity-aware multicenter evaluation of synthetic tabular ICU data

This repository contains the reproducible analysis code for evaluating generation reliability, representation fidelity, predictive utility, stochastic variability, multicenter heterogeneity, and empirical privacy of synthetic tabular ICU data.

## Study architecture

MIMIC-IV is the source database for model training and held-out internal evaluation. SICdb is an independent external evaluation database, and eICU-CRD provides multicenter external evaluation. Three generators (GaussianCopula, CTGAN, and TabDDPM) were evaluated across 15 prespecified seeds (42–56), with logistic regression and XGBoost as downstream learners. AUROC external utility loss (EUL) is the primary external estimand; transport interaction (ITL) is secondary. The multicenter analysis uses crossed seed and hospital effects with seed×hospital interaction and full within-hospital seed covariance.

The generation-reliability denominator contains all 45 attempts. Utility analyses are conditional on estimability: 15 GaussianCopula, 15 CTGAN, and 12 TabDDPM realizations. The three single-class TabDDPM collapses remain reliability outcomes and are not replaced or repaired.

## Data are not included

No row-level clinical data or synthetic datasets are included in this repository. MIMIC-IV, SICdb, and eICU-CRD must be obtained independently under their applicable access terms. The evaluated MIMIC-IV-derived row-level synthetic datasets are not redistributed under the applicable source-data governance restrictions. Aggregate results and fabricated demonstration materials are archived at https://doi.org/10.5281/zenodo.22822482.

See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) before running any source-data step.

## Repository map

- `configs/`: frozen study, generator, learner, bootstrap, hospital, and privacy settings.
- `src/cohort/`: MIMIC-IV cohort and source-window extraction.
- `src/harmonization/`: MIMIC-IV/SICdb static representation, training contract, and corrected eICU-CRD contract.
- `src/generators/`: authoritative SDV and non-EMA TabDDPM generation scripts.
- `src/models/`: exact learner kernels.
- `src/estimands/` and `src/bootstrap/`: utility metrics, calibration, and hierarchical inference.
- `src/multicenter/`: outcome-independent domain distance and full-covariance crossed-effects REML.
- `src/fidelity/`, `src/measurement_process/`, and `src/privacy/`: separate qualification-domain kernels.
- `src/reporting/`: editable Figure 1 generation.
- `tests/`: fabricated-data unit tests only.

## Reproduction boundary

Published results were generated under frozen software/data configurations. Reproduction requires access to the same source database releases and compatible software versions. The repository documents exact recovered versions where available, but does not promise bit-for-bit identity across platforms. Patient-level prediction files, fitted models, checkpoints, bootstrap indices, and aggregate manuscript outputs are deliberately excluded.

Start with [docs/ANALYSIS_WORKFLOW.md](docs/ANALYSIS_WORKFLOW.md), then review [docs/ESTIMANDS.md](docs/ESTIMANDS.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

For the patient-data-free, CPU-only post-fit qualification prototype and fully
fabricated reuse demonstration, see [README_PORTABLE.md](README_PORTABLE.md).

## Citation and license

Original software and accompanying documentation in this repository are licensed under the MIT License; see LICENSE. Third-party dependencies retain their own licenses and are not relicensed by this repository. This software license grants no rights to MIMIC-IV, SICdb, eICU-CRD, or restricted patient-derived data. The separate Zenodo materials retain the terms stated in that archive. The manuscript DOI is not yet assigned.
