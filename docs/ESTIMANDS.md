# Estimands and metric direction

For a performance metric where larger values are preferable:

\[
IUL=M_{RI}-M_{SI},\qquad EUL=M_{RE}-M_{SE},\qquad ITL=EUL-IUL.
\]

`RI` and `SI` denote real- and synthetic-trained models evaluated internally; `RE` and `SE` denote the corresponding models evaluated externally. AUROC EUL is the primary external estimand. AUROC ITL is the secondary transport interaction. Negative ITL indicates attenuation of the synthetic–real gap externally; it does not establish superior transportability or superior absolute external performance.

Skill transformations use a null reference calculated separately within each evaluation database:

\[
APS=(AP-\pi)/(1-\pi),
\]

\[
BSS=1-BS/[\pi(1-\pi)],
\]

\[
LLS=1-LL/[-\pi\log(\pi)-(1-\pi)\log(1-\pi)].
\]

Here \(\pi\) is the database-specific event prevalence. Calibration intercept is estimated as an offset model on clipped logits; calibration slope uses a freely estimated intercept and slope. No recalibration is performed.

The generator-level hierarchical bootstrap resamples synthetic seeds and the clinical evaluation sample. For eICU-CRD, hospitals are resampled and patients are resampled within selected hospitals. The crossed model is

\[
EUL_{sh}=\mu+u_s+v_h+w_{sh}+e_{sh},
\]

with full within-hospital seed×seed sampling covariance. Prediction intervals are reported for the hospital-average-seed target and for a new-seed/new-hospital target. Hospital-level ITL crossed inference was not estimated; the full joint MIMIC-plus-hospital covariance was not replaced by a simplified approximation.

The outcome-independent hospital distance \(D_W\) is the mean of 21 median-feature 1-Wasserstein distances, each scaled by the corresponding real MIMIC-IV training sample standard deviation (`ddof=1`).
