# Fully fabricated 8 × 12 demonstration fixture

This fixture is a transparent software-reuse example. It crosses eight identifiers
`FSEED_01`–`FSEED_08` with twelve identifiers `FSITE_01`–`FSITE_12`.

Each cell was generated deterministically as:

```text
EUL = 0.120 + fabricated seed effect + fabricated site effect
      + 0.020 × (((seed_index + 2 × site_index) mod 5) − 2)
```

The seed and site effects are listed implicitly by the released cells and generated
by the public fixture-build record. Each known covariance block is compound
symmetric: diagonal variance starts at `0.000225` and increases by `0.000005` per
site; off-diagonal covariance is 20% of that site's diagonal. Every 8 × 8 block is
positive definite.

Provenance:

```text
FIXTURE_CONTAINS_REAL_CLINICAL_DATA = NO
FIXTURE_DERIVED_FROM_REAL_PATIENT_ROWS = NO
FIXTURE_DERIVED_FROM_EMPIRICAL_HOSPITAL_ESTIMATES = NO
FIXTURE_PUBLIC_RELEASE_SAFE = YES
```

The fixture is fabricated, not deidentified. Its identifiers and values do not
represent hospitals, patients, or empirical ICU estimates.

The expected files freeze one execution of the existing full-covariance crossed
REML architecture and its deterministic qualification translation. The
REML/end-to-end absolute comparison tolerance is `1e-6`; pure algebra uses `1e-12`;
the variance-equivalence count matches exactly.

The demonstration command supplies `--delta 0.15` explicitly as a pedagogical
scalar. It is not a default and has no clinical interpretation.

```bash
.venv-portable/bin/python -m portable_qualification demo \
  --fixture portable_demo/example \
  --m-max 20 \
  --delta 0.15 \
  --output-dir portable_demo_output
```

Canonical calibration boundary:

> Calibration was target and distribution dependent: unseen-hospital generator-average coverage was less stable in some interaction-dominant and t5 settings. Universal calibration was not established, and intermediate-m outputs are model-derived rather than separately calibrated.

