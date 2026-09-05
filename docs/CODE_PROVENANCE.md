# Code provenance and bounded release edits

This release was curated from the locked project tree without rerunning analysis. Core production sources map as follows:

| Release path | Authoritative project source |
|---|---|
| `src/cohort/build_mimic_cohort.py` | `scripts/preflight1/mimic_01_cohort.py` |
| `src/cohort/extract_mimic_window.py` | `scripts/preflight1/mimic_02_extract_window.py` |
| `src/harmonization/build_static_representation.py` | `scripts/preflight2a/p2a_01_feature_extract.py` |
| `src/harmonization/build_training_contract.py` | `scripts/preflight2b0/b0_06_build_qualification_dataset.py` |
| `src/harmonization/build_eicu_contract.py` | `scripts/preflight2d/d09_extraction.py` plus the locked U0-R1 floor-hour density correction |
| `src/generators/run_sdv.py` | `scripts/preflight2b2/run_formal_sdv.py` |
| `src/generators/run_tabddpm.py` | `scripts/preflight2b2/run_formal_tabddpm.py` |
| `src/estimands/multidimensional_metrics.py` | `scripts/postreview_upgrade/u1f_multidimensional_utility/02_point_metrics.py` |
| `src/bootstrap/generator_hierarchical_bootstrap.py` | `scripts/postlock_upgrade/final_estimand_stochasticity_v1/u2_generator.py` |
| `src/multicenter/crossed_hospital_reml.py` | `scripts/postlock_upgrade/final_estimand_stochasticity_v1/u2_hospital.py` |
| `src/multicenter/build_domain_distance.py` | `scripts/postreview_upgrade/u1h_b12_protocol_conformant_repair_r1/01_build_dw_preoutcome.py` |
| `src/reporting/generate_figure1.py` | `scripts/postlock_upgrade/v5r2_figure1_reconstruction/generate_figure1_v5r2.py` |

Bounded edits replaced local absolute paths with environment variables or repository-relative paths, removed internal-document provenance pointers, extended frozen generator config seed lists from the original five-seed files to the authorized 42–56 universe, and folded the locked `FLOOR(offset/60)` density semantics into the eICU extraction copy. No estimand, endpoint, model parameter, generator parameter, resampling rule, or crossed-effects definition was changed.

Small importable kernels for frozen learners, utility formulas, fidelity, measurement density, and empirical attacks were extracted verbatim in mathematical behavior from their production scripts to support review and fabricated-data tests. Historical orchestration that depended on private checkpoints, protected manifests, or internal reports is intentionally not distributed.
