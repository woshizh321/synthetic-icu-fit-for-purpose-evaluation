# E1 public decision translation and verification

This directory provides the frozen P95 three-state decision translation and a verifier for the released E1 aggregate exports. It does not redistribute row-level MIMIC-IV, SICdb, or eICU data; evaluated synthetic patient rows; patient predictions; hospital identifiers; fitted clinical models; covariance assets; or private bootstrap indices.

Run from this directory:

```sh
python verify_released_e1.py
```

The verifier checks the five released E1 aggregate files against their locked source hashes, independently translates all released E1 decision probabilities, and reconstructs the 24 M4 decision-count rows from the released realization-decision records.

This is aggregate verification and decision-layer reproduction. Regenerating E1 from clinical source records requires independently authorized access to MIMIC-IV v3.1, SICdb v1.0.8, and eICU-CRD v2.0, plus restricted prediction, hospital-indexed, covariance, and bootstrap dependencies that are deliberately excluded from the public release.
