# Reproduction boundary and commands

Public execution has four paths: fabricated-data software tests, deterministic historical V2 display rendering, deterministic R2 figure rendering, and extension verification/reproduction. Install `requirements-display.txt` for plotting, `requirements-analysis.txt` for E3/E5, and `requirements-test.txt` for tests. The release audit records actual test counts and the tested environment; no historical test count is a current guarantee.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q tests

python -m pip install -r requirements-display.txt
python reproduction/reproduce_main_displays.py --output reproduced_displays
python reproduction/verify_public_release.py --rendered reproduced_displays
python reproduction/reproduce_r2_figures.py --output-dir reproduced_r2_figures

python extension/e1/verify_released_e1.py

python -m pip install -r requirements-analysis.txt
python extension/e3_e5/reproduce_e3_e5.py --output-dir reproduced_extension
```

The manuscript source CSVs retain their existing numeric strings. Historical display files and final R2 Figure 1–5 files are preserved. Rendering adapts only filesystem/argument plumbing and preserves the executed display geometry and encodings. R2 PNGs are required to match the released figures pixel-for-pixel; PDF metadata or generated object identifiers can differ across runs.

The E1 public verifier checks five authoritative aggregate hashes, independently applies the frozen P95 rule to all released decision rows, and reconstructs the M4 count summary. It does not regenerate clinical predictions, EUL cells, variance components, or covariance estimates.

The E3/E5 public input contains 1,152,000 simulated replicate-strategy records and preserves all 812 failed fits. The public script reconstructs the full 5 × 6 cost grid, E3 scenario-conditional regret, paired comparisons, fixed-P95 sensitivity, and E5 leave-one-generator-out results. It verifies outputs against Supplementary Data 6–10 and checks the E5 54/31/71 counts. It performs no simulation, fitting, prediction, or covariance estimation.

Current scientific kernels preserve likelihoods, optimizer starts/bounds/tolerances, decision thresholds, variance standardization and deterministic stream derivation. Input/output paths and imports are adapted. The fixed random-stream namespace is retained as a reproducibility constant; it is not a release-history statement.

Running new known-truth simulations requires empirical calibration templates and sampling covariance interfaces from an authorized reconstruction; those restricted hospital-level inputs are not included. The released minimal simulated records support E3/E5 replay only. The public release does not promise exact reproduction of historical clinical extraction, splits, predictions, model training, E1 variance fitting, or covariance estimation. Bootstrap predictive mixtures condition on supplied estimated sampling covariance and do not propagate uncertainty in that covariance itself.

The four tolerances and multiple DGPs are distinct related scenarios. No composite scientific success rule, strategy optimization or calibration certification is supplied.
