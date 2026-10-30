# E3 and E5 public reproduction

`public_inputs/V2_P95_REPLICATE_METRICS.parquet` contains the minimal simulated replicate-level P95 quantities needed to reproduce the locked E3 scenario-conditional cost-sensitive regret and E5 leave-one-generator-out analyses. It contains no clinical records, patient predictions, hospital identifiers, evaluated synthetic patient rows, fitted models, or covariance assets.

Run from the repository root:

```sh
python extension/e3_e5/reproduce_e3_e5.py --output-dir reproduced_extension
```

The script preserves the frozen 5 × 6 cost grid, complete three-action Bayes argmin, failed-fit accounting, common-success pairing, equal macro weights, and fixed-P95 sensitivity. It verifies every emitted table against Supplementary Data 6–10 and verifies the E5 counts 54 direction changes, 31 interval-supported reversals, and 71 interval-classification changes. It performs no simulation, model fitting, prediction, covariance estimation, or clinical-data access.

E3 remains conditional on the 24 original known-truth scenarios per DGP × tolerance composition. It does not supply a continuous latent-probability sweep or support claims of universal strategy optimality.
