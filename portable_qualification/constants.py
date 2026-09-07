"""Frozen public constants for evidence-card version 1."""

SCHEMA_VERSION = "1.0.0"
QUALIFICATION_MODEL = "GAUSSIAN_CROSSED_RANDOM_EFFECTS_PLUGIN"
SOFTWARE_VERSION = "portable-qualification-1.0.0"
__version__ = "1.0.0"

QUALIFICATION_BOUNDARY = (
    "This is a Gaussian random-effects plug-in probability derived from the "
    "fitted crossed model. It is not a clinical guarantee, certification "
    "probability, or patient-benefit estimate."
)

# Exact U5-00 canonical machine-readable boundary. Keep one source of truth.
U4_CALIBRATION_BOUNDARY = (
    "Calibration was target and distribution dependent: unseen-hospital "
    "generator-average coverage was less stable in some interaction-dominant "
    "and t5 settings. Universal calibration was not established, and "
    "intermediate-m outputs are model-derived rather than separately calibrated."
)

INTERMEDIATE_M_STATUS = "MODEL_DERIVED_EXPLORATORY"
INTERMEDIATE_M_BOUNDARY = (
    "Only the m=1 and m→∞ limits are directly linked to the previously evaluated "
    "Target B and Target A definitions. Intermediate m values are deterministic "
    "extensions of the fitted variance model and were not separately calibrated in U4."
)

M_EQ_LABEL = "variance-equivalence realization count"
M_EQ_DEFINITION = (
    "Minimum integer number of independent stochastic realizations for which "
    "residual generator-related variance is no greater than the estimated "
    "hospital variance."
)
M_EQ_BOUNDARY = (
    "This is a fitted variance-component design quantity, not an optimal, "
    "recommended, required, or clinically sufficient realization count."
)

PROHIBITED_CARD_FIELDS = {
    "pass",
    "fail",
    "safe",
    "unsafe",
    "certified",
    "approved",
    "clinically_acceptable",
    "recommended",
}
