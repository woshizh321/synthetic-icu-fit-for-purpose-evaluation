"""Attack-specific empirical privacy diagnostics; no formal privacy guarantee."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def distance_diagnostics(real_train_scaled, real_test_scaled, synthetic_scaled, seed: int = 42):
    rng = np.random.RandomState(seed)
    reference_idx = rng.choice(len(real_test_scaled), 2000, replace=False)
    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(real_train_scaled)
    reference_distances, _ = real_nn.kneighbors(real_test_scaled[reference_idx])
    reference_dcr_p5 = float(np.percentile(reference_distances[:, 0], 5))

    synthetic_distances, _ = real_nn.kneighbors(synthetic_scaled)
    dcr = synthetic_distances[:, 0]
    nndr = dcr / np.clip(synthetic_distances[:, 1], 1e-12, None)
    near = (nndr < 0.1) & (dcr < reference_dcr_p5)
    return {
        "exact_duplicate_count": int(np.sum(dcr <= 1e-12)),
        "near_duplicate_count": int(near.sum()),
        "near_duplicate_rate": float(near.mean()),
        "dcr": dcr,
        "nndr": nndr,
    }


def membership_inference(real_train_scaled, real_test_scaled, synthetic_scaled,
                         seed: int = 42, bootstrap_replicates: int = 200):
    rng = np.random.RandomState(seed)
    # Preserve the frozen shared RNG sequence: the reference-test draw preceded
    # the member and nonmember draws in the production implementation.
    rng.choice(len(real_test_scaled), 2000, replace=False)
    member_idx = rng.choice(len(real_train_scaled), 2000, replace=False)
    nonmember_idx = rng.choice(len(real_test_scaled), 2000, replace=False)
    attack_y = np.r_[np.ones(2000), np.zeros(2000)]
    nearest_synthetic = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(synthetic_scaled)
    member_distance, _ = nearest_synthetic.kneighbors(real_train_scaled[member_idx])
    nonmember_distance, _ = nearest_synthetic.kneighbors(real_test_scaled[nonmember_idx])
    score = -np.r_[member_distance[:, 0], nonmember_distance[:, 0]]
    auc = float(roc_auc_score(attack_y, score))
    bootstrap_rng = np.random.RandomState(seed)
    boot = []
    for _ in range(bootstrap_replicates):
        indices = bootstrap_rng.randint(0, len(attack_y), len(attack_y))
        boot.append(roc_auc_score(attack_y[indices], score[indices]))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return {"AUC": auc, "CI_lower": float(lower), "CI_upper": float(upper),
            "advantage": float(2 * auc - 1), "formal_privacy_guarantee": False}
