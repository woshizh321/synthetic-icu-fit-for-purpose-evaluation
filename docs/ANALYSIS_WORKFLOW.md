# Analysis and reproduction workflow

1. Inspect current definitions in ESTIMANDS.md and the released study config.
2. Run public fabricated tests; these use no clinical database.
3. Verify approved aggregate sources and final manuscript figures against the public manifest.
4. Render Figures 1–5 from released aggregate CSVs. Rendering performs no fitting, new decision calculation or scientific resampling.
5. For separately authorized deeper reconstruction, supply the input contracts described in DATA_ACCESS.md. Source preparation/training code remains an access-dependent interface, not evidence that all historical raw-source steps are independently replayable.

`decision_evaluation/core.py` contains the executed M1/M2 crossed REML engine, hospital-only estimators, standardized Gaussian/t5/skew-normal DGPs, frozen decision rules and metrics. `simulation.py` contains deterministic paired M1–M4 orchestration and latent truth generation. `summarize.py` contains equal-cell summaries, common-success paired contrasts, coverage and recovery. `bootstrap.py` implements conditional-on-sampling-covariance uncertainty; `operational.py` separates generation reliability from conditional utility. These modules are provided for inspection and authorized reconstruction, and are never invoked by display reproduction.

The portable fabricated-data interface is a bounded demonstration with its own frozen compatibility fixtures. It is not an alternative authority for current manuscript estimates. General clinical-pipeline utilities retain schema-level identifiers in code; no records containing their values are released.
