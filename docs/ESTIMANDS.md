# Estimands and decisions

EUL = destination AUROC(real-trained) − destination AUROC(synthetic-trained), with learner and destination held fixed. Primary Target B is a new utility-estimable realization at a future H200-exchangeable hospital. Secondary Target A averages across realizations at a future eligible hospital.

For M1/M2 the plug-in predictive variance is Var(mu-hat)+v_h for A and Var(mu-hat)+v_s+v_h+v_interaction for B. Future AUROC measurement error is not added. M3/M4 use a hospital-only distribution and therefore have naive Target-B extrapolations. M4's A output is limited to a specific realization.

P95 uses probabilities ≥.95 / ≤.05 for qualified / not-qualified; all other successful fits are indeterminate. C95 qualifies when q97.5≤delta and rejects when q2.5>delta. Failure is separate from all three decision states. Confidence sensitivity uses .80,.90,.95,.975,.99 at each fixed delta.

DER is mismatch probability against the three-state oracle; DY is definitive decision yield; CDY is correct definitive yield. FQ-all is false qualification over all successful fits. SER and FQ-def condition on definitive decisions within a template–regime cell. Their macro estimates average defined cell ratios, omitting cells with zero definitive count and explicitly reporting contributing cell_n. They are not ratios of pooled counts. Paired contrasts use common successful replicates; four-way-common sensitivity uses the intersection of all four strategies.

The public helper API `src.estimands.utility` also exposes internal/external performance-difference algebra with API keys IUL and ITL for compatibility. These helper quantities are not the manuscript's future-use targets or a secondary transport claim. Current future-use inference is implemented in `decision_evaluation/`.
