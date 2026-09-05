# Reproducibility and software environment

Two production environments were used and are recorded separately.

## Generator environment (exact recovered core)

- Python 3.11.15
- PyTorch 2.6.0+cu124
- CUDA runtime reported by PyTorch: 12.4
- SDV 1.38.0
- TabDDPM upstream commit: `b476257dd460b778ba09eb97f7a51d6490fa17f8`

Hardware and host names are not required scientific inputs and are omitted. Versions of transitive packages not present in the frozen runtime records are unavailable rather than inferred.

## Prediction environment (exact recovered)

- Python 3.14.4
- NumPy 2.5.2
- pandas 3.0.5
- scikit-learn 1.9.0
- XGBoost 3.4.0
- PyArrow 25.0.1

## Fidelity/calibration and cohort records

The fidelity/privacy runtime recorded Python 3.14.4, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, and Matplotlib 3.10.9. The eICU endpoint feasibility runtime recorded DuckDB 1.5.3 and PyArrow 24.0.0. These differences are historical facts; no single invented lock file is presented as the environment for every stage.

## Boundaries

- The source database releases and local source infrastructure are external dependencies.
- Historical fitted objects and patient-level predictions are excluded.
- Exact GPU execution can depend on platform-specific numerical behavior.
- All 45 attempts belong to reliability; conditional utility uses the 42 estimable datasets.
- Empirical attacks are attack-specific diagnostics and do not provide formal privacy accounting.
- Syntax and fabricated-data unit tests do not validate source-data access or reproduce manuscript numbers.
