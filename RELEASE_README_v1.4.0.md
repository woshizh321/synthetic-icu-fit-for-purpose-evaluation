# v1.4.0 R2 public reproducibility synchronization

This immutable release synchronizes the completed R2 manuscript displays and Supplementary Data with the public reproduction package while preserving the historical v1.3.0 release.

## Added

- Final five R2 main figures and aggregate figure inputs.
- Supplementary Data 1–10 and the final R2 Supplement PDF.
- Public E1 P95 decision-translation code and locked aggregate-hash verification.
- A minimal simulated replicate-level P95 input and executable E3/E5 reproduction.
- Exact verification of the E5 54 direction changes, 31 interval-supported reversals, and 71 interval-classification changes.
- Relative-path R2 figure reproduction with pixel-identical PNG verification.

## Reproduction boundary

Original V2 aggregate displays, R2 figures, E1 aggregate decision translation, and E3/E5 are reproducible from released inputs. Full E1 clinical-source regeneration requires independently authorized MIMIC-IV v3.1, SICdb v1.0.8, and eICU-CRD v2.0 access plus restricted prediction, hospital-indexed, covariance, and bootstrap dependencies.

No patient-level clinical or evaluated synthetic data, patient predictions, hospital identifiers, fitted clinical models, restricted hospital-indexed derivatives, private covariance assets, resampling indices, credentials, or internal project records are included.

Software is MIT licensed. Released aggregate/simulated data, figures, Supplementary Data and documentation are CC BY 4.0. Upstream databases and third-party dependencies retain their own terms.

Stable archive series: Zenodo concept DOI [10.5281/zenodo.22822481](https://doi.org/10.5281/zenodo.22822481). The version-specific DOI is added to the GitHub release after archive publication.
