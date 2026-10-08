# Task-specific synthetic-data suitability across clinical destinations

Corrected, scope-limited public research software and aggregate publication data. Read [reproducibility boundaries](docs/REPRODUCIBILITY.md), [code provenance](docs/CODE_PROVENANCE.md) and [data access](docs/DATA_ACCESS.md) before interpreting any reproducibility claim.

## Install and test

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-public.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python -m public_producers.fixture --output /tmp/icu-invented-demo-new-directory
```

The fixture uses invented records, supplied probabilities and supplied Target-B intervals, covering three generator labels, two learners, 85 invented destinations and four tolerances. It trains no clinical model and refits no clinical crossed model. Working fixture identifiers are invented. Its public aggregate outputs test software, not original clinical preprocessing.

## Included materials

Publication data contain only approved 05S aggregate projections and supplementary workbooks. Tests verify frozen aggregate source identity, decision/discordance computations on fabricated data, restricted-field rejection, the corrected TabDDPM 16,000-update linear optimizer schedule separately from cosine diffusion, and eICU export-path handling. Clinical generation modules remain inspectable but are not executed by this demonstration; their optional dependencies are separate.

Historical MIMIC membership is internally verified but exact public raw-source replay is not established. Canonical eICU-to-analytical input equivalence is verified; original raw CSV-to-canonical conversion is not independently verified and historical D-09 executed source remains unresolved. Reference and publication-interface code must not be called authenticated original source.

Clinical databases require independent source authorization. No patient-level clinical records/predictions, evaluated row-level synthetic data, fitted clinical models/checkpoints, source hospital identifiers, hospital-indexed outputs, private maps, restricted bootstrap or historical split assets are distributed. MIT software licensing grants no clinical-data access rights. Published summary research findings retain their release-specific attribution and source governance.
