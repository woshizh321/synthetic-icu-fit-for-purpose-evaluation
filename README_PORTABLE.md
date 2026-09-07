# Portable post-fit qualification

## Scientific purpose

This CPU-only prototype translates aggregate outputs from a Gaussian crossed
seed-by-hospital model into realization-number-dependent uncertainty summaries.
It treats stochastic generator realization and destination hospital as distinct
sources of modeled heterogeneity. It requires no patient rows, predictions,
synthetic clinical datasets, or GPU.

The tool calculates (V_{het}(m)), (V_{pred}(m)), (G(m)), the
**variance-equivalence realization count**, and—only when the user explicitly
supplies `delta`—a **model-based fit-for-purpose tolerance probability**.

It does not define a clinical tolerance, certify a dataset, establish clinical
acceptability, rerun an empirical analysis, or provide a formal privacy guarantee.
No default `delta` is used.

## Installation from a fresh clone

The commands below are complete and assume only Git and Python 3.11 or newer are available.
Run them from an empty working directory:

```bash
git clone https://github.com/woshizh321/synthetic-icu-fit-for-purpose-evaluation.git
cd synthetic-icu-fit-for-purpose-evaluation
python3 -m venv .venv-portable
.venv-portable/bin/python -m pip install --upgrade pip
.venv-portable/bin/python -m pip install -r requirements-portable.txt
```

No shell alias, local project path, manual file edit, clinical database, or GPU is
required.

## Mode A: post-fit qualification

Provide a JSON object or one-row CSV summary containing the version-1 aggregate inputs documented in
`schema/evidence_card.schema.json`, including all reliability counts, the fitted
mean and standard error, three variance components, existing Target-A and Target-B
interval endpoints, and immutable source identity.

Without a user tolerance:

```bash
.venv-portable/bin/python -m portable_qualification qualify \
  --summary portable_demo_output/fit_summary.json \
  --m-max 20 \
  --output-dir portable_qualification_output
```

With an explicitly user-supplied pedagogical tolerance:

```bash
.venv-portable/bin/python -m portable_qualification qualify \
  --summary portable_demo_output/fit_summary.json \
  --m-max 20 \
  --delta 0.15 \
  --output-dir portable_qualification_output_with_delta
```

Outputs are `evidence_card.json`, a flattened `evidence_card.csv`,
`realization_budget.csv`, and `qualification_metadata.json`. When `--delta` is
absent, both the input tolerance and probability map remain JSON null.

## Mode B: fully fabricated demonstration

The public fixture contains 8 fabricated stochastic realizations crossed with 12
fabricated sites and positive-definite fabricated covariance blocks. Run the exact
end-to-end demonstration from the repository root:

```bash
.venv-portable/bin/python -m portable_qualification demo \
  --fixture portable_demo/example \
  --m-max 20 \
  --delta 0.15 \
  --output-dir portable_demo_output
```

Expected output files:

- `portable_demo_output/fit_summary.json`
- `portable_demo_output/evidence_card.json`
- `portable_demo_output/evidence_card.csv`
- `portable_demo_output/realization_budget.csv`
- `portable_demo_output/qualification_metadata.json`

The expected numerical record is packaged in
`portable_demo/example/fixture_expected_fit.json` and
`portable_demo/example/fixture_expected_qualification.json`. REML/end-to-end
comparisons use the prospectively specified absolute tolerance `1e-6`; algebraic
self-consistency uses `1e-12`; `m_eq` matches exactly.

The `fit` subcommand can also be invoked separately:

```bash
.venv-portable/bin/python -m portable_qualification fit \
  --cells portable_demo/example/fixture_cells.csv \
  --covariance portable_demo/example/fixture_sampling_covariance.csv \
  --reliability portable_demo/example/fixture_reliability.json \
  --output portable_demo_output/fit_summary.json
```

## Evidence-card fields

The JSON card records generator and learner identities, generation-reliability
counts, mean EUL and its standard error, the three fitted variance components,
existing Target-A and Target-B intervals, `m_eq`, the realization grid,
`V_het`, `V_pred`, `G`, optional tolerance probabilities, scientific boundaries,
source hashes, schema version, and software version. The JSON Schema rejects
additional decision-label fields.

## Interpretation boundaries

The **variance-equivalence realization count** is the minimum integer number of
independent stochastic realizations for which residual generator-related variance
is no greater than the estimated hospital variance. This is a fitted
variance-component design quantity, not an optimal, recommended, required, or
clinically sufficient realization count.

Only the `m=1` and analytical `m→∞` limits are directly linked to the previously
evaluated Target B and Target A definitions. Intermediate values are deterministic
extensions of the fitted variance model and were not separately calibrated.

Canonical calibration boundary:

> Calibration was target and distribution dependent: unseen-hospital generator-average coverage was less stable in some interaction-dominant and t5 settings. Universal calibration was not established, and intermediate-m outputs are model-derived rather than separately calibrated.

An explicitly supplied `delta` is a use-specific upper bound chosen by the user.
The returned quantity is a Gaussian random-effects plug-in probability. It is not a
clinical guarantee, certification probability, or patient-benefit estimate.

## Data, privacy, hardware, and license

The fixture is fabricated rather than deidentified. It contains no real clinical
data, patient rows, empirical hospital estimates, or clinical site identifiers.
The portable workflow is CPU-only and imports no GPU framework.

The software license decision is pending PI determination. The code is publicly
accessible, but no reuse license or “open source” status is implied.

## Technical tests

From the activated environment:

```bash
.venv-portable/bin/python -m unittest tests.test_portable_qualification -v
```

These tests cover the locked algebra, edge cases, schema policy, fabricated REML
fit, Target A/B bridge, and end-to-end expected outputs. Test success is a technical
QC result, not a scientific qualification decision.
