# Public release audit v1.4.0

## Scope

The release is additive to v1.3.0. It adds the R2 displays, Supplementary Data 1–10, public E1 aggregate decision verification, and replay of the locked E3/E5 analyses from a minimal simulated replicate-level input. It performs no new clinical analysis, simulation, model fitting, prediction, covariance estimation, E2, or E4.

## Executed verification

- Original public tests: 36 tests and 15 subtests passed.
- Original V2 aggregate source and display checks: passed; five main PNGs reproduced pixel-for-pixel.
- E1: five locked aggregate hashes matched; 496 released P95 decision rows retransformed; 24 M4 count rows reconstructed.
- E3/E5: all five output tables reproduced and matched Supplementary Data 6–10; E5 counts 54/31/71 passed.
- R2 figures: all five PNGs reproduced pixel-for-pixel from released aggregate inputs.
- Supplementary Data: all ten file hashes matched the R2 source ledger.

## Privacy and licensing

The proposed tree, packaged assets, PDF text, Parquet schema, filenames, and pushed Git history were scanned before publication. No credentials, private keys, local absolute paths, patient or encounter identifiers, source hospital identifiers, patient predictions, row-level clinical data, evaluated synthetic patient rows, fitted clinical models, or restricted covariance/bootstrap assets are included. The E3/E5 Parquet is simulated and contains 1,152,000 replicate-strategy rows, 17 declared nonclinical columns, and all 812 failed-fit statuses.

Original software is MIT licensed. Released aggregate/simulated data, figures, Supplementary Data and documentation are CC BY 4.0. Clinical databases and their restricted derivatives are excluded and retain their own access terms.

## Scientific boundary

E1 is aggregate decision-layer verification, not public raw-to-result clinical replay. E3 is conditional on the original fixed known-truth scenarios and does not provide a continuous latent-probability sweep. E5 is a benchmark-mixture sensitivity and does not establish generator training adequacy or causal generator effects.
