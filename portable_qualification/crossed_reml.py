"""CPU-only full-covariance crossed REML for the fabricated public demo.

The likelihood and optimizer settings match the existing public crossed-model
implementation. This module accepts only aggregate seed-by-hospital cells and
known within-hospital covariance blocks.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def fit_crossed_reml(
    cells: list[dict[str, Any]],
    covariance: dict[str, tuple[list[str], np.ndarray]],
) -> dict[str, Any]:
    if not cells:
        raise ValueError("cells must not be empty")
    ordered = sorted(cells, key=lambda row: (str(row["hospital_id"]), str(row["seed_id"])))
    hospitals = sorted({str(row["hospital_id"]) for row in ordered})
    seeds = sorted({str(row["seed_id"]) for row in ordered})
    if len(ordered) != len(hospitals) * len(seeds):
        raise ValueError("fixture must contain one complete cell for every seed-hospital pair")
    keys = [(str(row["hospital_id"]), str(row["seed_id"])) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed-hospital cells")
    seed_map = {seed: index for index, seed in enumerate(seeds)}
    y = np.asarray([float(row["EUL"]) for row in ordered], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("EUL values must be finite")
    x = np.ones((len(y), 1), dtype=float)
    blocks: list[tuple[slice, np.ndarray, np.ndarray]] = []
    offset = 0
    for hospital in hospitals:
        rows = [row for row in ordered if str(row["hospital_id"]) == hospital]
        block_seeds = [str(row["seed_id"]) for row in rows]
        if hospital not in covariance:
            raise ValueError(f"missing covariance block for hospital {hospital}")
        covariance_seeds, full = covariance[hospital]
        if set(covariance_seeds) != set(seeds):
            raise ValueError(f"covariance seeds do not match cells for hospital {hospital}")
        indices = [covariance_seeds.index(seed) for seed in block_seeds]
        known = np.asarray(full, dtype=float)[np.ix_(indices, indices)]
        if known.shape != (len(seeds), len(seeds)) or not np.isfinite(known).all():
            raise ValueError("invalid covariance block dimensions or values")
        if not np.allclose(known, known.T, atol=1e-12, rtol=0):
            raise ValueError("covariance block must be symmetric")
        if np.linalg.eigvalsh(known).min() <= 0:
            raise ValueError("covariance block must be positive definite")
        z_seed = np.zeros((len(block_seeds), len(seeds)))
        z_seed[np.arange(len(block_seeds)), [seed_map[seed] for seed in block_seeds]] = 1
        blocks.append((slice(offset, offset + len(block_seeds)), known, z_seed))
        offset += len(block_seeds)

    def solve(theta: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
        variance_seed, variance_hospital, variance_interaction = np.exp(theta)
        inverse_q = np.zeros_like(q)
        inverse_z = np.zeros((len(y), len(seeds)))
        log_determinant = 0.0
        for block_slice, known, z_seed in blocks:
            matrix = (
                known
                + variance_interaction * np.eye(len(known))
                + variance_hospital * np.ones_like(known)
            )
            factor = cho_factor(matrix, lower=True, check_finite=False)
            log_determinant += 2 * np.log(np.diag(factor[0])).sum()
            inverse_q[block_slice] = cho_solve(factor, q[block_slice], check_finite=False)
            inverse_z[block_slice] = cho_solve(factor, z_seed, check_finite=False)
        kernel = np.eye(len(seeds)) / variance_seed
        kernel += sum(
            (z_seed.T @ inverse_z[block_slice] for block_slice, _, z_seed in blocks),
            np.zeros((len(seeds), len(seeds))),
        )
        kernel_factor = cho_factor(kernel, lower=True, check_finite=False)
        zt_inverse_q = sum(
            (z_seed.T @ inverse_q[block_slice] for block_slice, _, z_seed in blocks),
            np.zeros((len(seeds), q.shape[1])),
        )
        corrected = inverse_q - inverse_z @ cho_solve(
            kernel_factor, zt_inverse_q, check_finite=False
        )
        log_determinant += len(seeds) * math.log(variance_seed)
        log_determinant += 2 * np.log(np.diag(kernel_factor[0])).sum()
        return corrected, float(log_determinant)

    def objective(theta: np.ndarray) -> float:
        q = np.column_stack([y, x])
        inverse_q, log_determinant = solve(theta, q)
        inverse_y = inverse_q[:, 0]
        inverse_x = inverse_q[:, 1:]
        information = x.T @ inverse_x
        try:
            beta = np.linalg.solve(information, x.T @ inverse_y)
        except np.linalg.LinAlgError:
            return 1e100
        residual = y - x @ beta
        inverse_residual, _ = solve(theta, residual[:, None])
        quadratic = float(residual @ inverse_residual[:, 0])
        sign, log_information = np.linalg.slogdet(information)
        if sign <= 0 or not np.isfinite(quadratic):
            return 1e100
        return 0.5 * (
            log_determinant
            + log_information
            + quadratic
            + (len(y) - x.shape[1]) * math.log(2 * math.pi)
        )

    starts = [
        np.log([1e-4, 1e-4, 1e-4]),
        np.log([1e-3, 1e-3, 1e-3]),
        np.log([1e-5, 1e-4, 1e-3]),
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(-27.63, 0)] * 3,
            options={"maxiter": 300, "ftol": 1e-11},
        )
        for start in starts
    ]
    optimum = min(fits, key=lambda result: result.fun)
    inverse_q, _ = solve(optimum.x, np.column_stack([y, x]))
    inverse_y = inverse_q[:, 0]
    inverse_x = inverse_q[:, 1:]
    variance_beta = np.linalg.inv(x.T @ inverse_x)
    beta = variance_beta @ (x.T @ inverse_y)
    se = np.sqrt(np.diag(variance_beta))
    variances = np.exp(optimum.x)
    return {
        "converged": bool(optimum.success),
        "optimizer_message": str(optimum.message),
        "reml_objective": float(optimum.fun),
        "mu_eul": float(beta[0]),
        "se_mu": float(se[0]),
        "sigma2_seed": float(variances[0]),
        "sigma2_hospital": float(variances[1]),
        "sigma2_seed_hospital": float(variances[2]),
        "hospital_n": len(hospitals),
        "seed_n": len(seeds),
        "cell_n": len(y),
        "method": "CUSTOM_EXACT_GAUSSIAN_REML_KNOWN_BLOCK_V",
    }
