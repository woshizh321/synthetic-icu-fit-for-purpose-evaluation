#!/usr/bin/env python3
"""Utilities for target-aware synthetic clinical-data evaluation.

This module contains no outcome-dependent tuning.  It implements the locked
random streams, latent distributions, decision policies, metrics, and the
authenticated M3/M4 hospital-only estimators used by the production scripts.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal, ROUND_CEILING
from typing import Iterable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.stats import norm

MASTER_SEED = 20261020
SEED_PREFIX = "ICU_SYNTH_V2"
Z975 = 1.959963984540054
DELTAS = (0.025, 0.05, 0.10, 0.20)
CONFIDENCE_GRID = (0.80, 0.90, 0.95, 0.975, 0.99)
DGPS = ("G", "T", "S")
METHODS = (
    "M1_FULL_COV_CROSSED",
    "M2_DIAGONAL_COV_CROSSED",
    "M3_REALIZATION_AGGREGATED",
    "M4_SINGLE_REALIZATION",
)
REGIMES = (
    "R1_SEED_DOMINANT",
    "R2_HOSPITAL_DOMINANT",
    "R3_INTERACTION_DOMINANT",
    "R4_MIXED",
)
STATES = ("QUALIFIED", "INDETERMINATE", "NOT_QUALIFIED")


def derive_seed(module: str, dgp: str, template: str, regime: str, replicate: int) -> int:
    """Derive one deterministic unsigned 64-bit seed from the locked key."""
    key = f"{SEED_PREFIX}|{module}|{dgp}|{template}|{regime}|{replicate}|{MASTER_SEED}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big", signed=False)


def standardized_draws(dgp: str, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw a zero-mean, unit-variance latent component under a locked DGP."""
    if dgp == "G":
        return rng.standard_normal(size)
    if dgp == "T":
        return rng.standard_t(5, size=size) * math.sqrt(3.0 / 5.0)
    if dgp == "S":
        alpha = 5.0
        delta = alpha / math.sqrt(1.0 + alpha * alpha)
        raw = delta * np.abs(rng.standard_normal(size)) + math.sqrt(1.0 - delta * delta) * rng.standard_normal(size)
        mean = delta * math.sqrt(2.0 / math.pi)
        sd = math.sqrt(1.0 - 2.0 * delta * delta / math.pi)
        return (raw - mean) / sd
    raise ValueError(f"Unknown DGP: {dgp}")


def draw_component(dgp: str, variance: float, size: int | tuple[int, ...], seed: int) -> np.ndarray:
    if variance < 0 or not math.isfinite(variance):
        raise ValueError("variance must be finite and nonnegative")
    if variance == 0:
        return np.zeros(size, dtype=float)
    return standardized_draws(dgp, size, np.random.default_rng(int(seed))) * math.sqrt(variance)


def predictive_probability(delta: float, mu: float, sd: float) -> float:
    if sd < 0 or not math.isfinite(sd):
        return math.nan
    if sd == 0:
        return float(mu <= delta)
    return float(norm.cdf((delta - mu) / sd))


def probability_state(p: float, confidence: float = 0.95) -> str:
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if p >= confidence:
        return "QUALIFIED"
    if p <= 1.0 - confidence:
        return "NOT_QUALIFIED"
    return "INDETERMINATE"


def interval_state(q025: float, q975: float, delta: float) -> str:
    if q975 <= delta:
        return "QUALIFIED"
    if q025 > delta:
        return "NOT_QUALIFIED"
    return "INDETERMINATE"


def classify_draws(draws: np.ndarray, delta: float, policy: str, confidence: float | None = None) -> tuple[str, float, float, float]:
    draws = np.asarray(draws, float)
    p = float(np.mean(draws <= delta))
    q025, q975 = np.quantile(draws, [0.025, 0.975])
    if policy == "P95":
        state = probability_state(p, 0.95)
    elif policy == "C95":
        state = interval_state(float(q025), float(q975), delta)
    elif policy == "RISK_YIELD":
        if confidence is None:
            raise ValueError("confidence required for risk-yield classification")
        state = probability_state(p, confidence)
    else:
        raise ValueError(policy)
    return state, p, float(q025), float(q975)


def decision_flags(oracle: str, estimated: str) -> dict[str, int]:
    if oracle not in STATES or estimated not in STATES:
        raise ValueError("unknown decision state")
    definitive = estimated != "INDETERMINATE"
    mismatch = estimated != oracle
    return {
        "decision_error": int(mismatch),
        "estimated_definitive": int(definitive),
        "selective_error": int(definitive and mismatch),
        "false_qualification": int(estimated == "QUALIFIED" and oracle != "QUALIFIED"),
        "false_rejection": int(estimated == "NOT_QUALIFIED" and oracle != "NOT_QUALIFIED"),
        "correct_definitive": int(definitive and not mismatch),
        "under_recognition_uncertainty": int(definitive and oracle == "INDETERMINATE"),
        "excess_indeterminacy": int(estimated == "INDETERMINATE" and oracle != "INDETERMINATE"),
    }


def summarize_decisions(rows: Iterable[dict]) -> dict[str, float | int]:
    records = list(rows)
    n = len(records)
    if n == 0:
        return {k: math.nan for k in ["DER", "DY", "SER", "FQ_all", "FQ_def", "FR_all", "CDY", "under_recognition", "excess_indeterminacy"]} | {"evaluable_n": 0, "definitive_n": 0}
    flags = [decision_flags(r["oracle_state"], r["estimated_state"]) for r in records]
    sums = {k: sum(f[k] for f in flags) for k in flags[0]}
    dn = sums["estimated_definitive"]
    return {
        "evaluable_n": n,
        "definitive_n": dn,
        "DER": sums["decision_error"] / n,
        "DY": dn / n,
        "SER": sums["selective_error"] / dn if dn else math.nan,
        "FQ_all": sums["false_qualification"] / n,
        "FQ_def": sums["false_qualification"] / dn if dn else math.nan,
        "FR_all": sums["false_rejection"] / n,
        "CDY": sums["correct_definitive"] / n,
        "under_recognition": sums["under_recognition_uncertainty"] / n,
        "excess_indeterminacy": sums["excess_indeterminacy"] / n,
    }


def fit_hospital(effect: np.ndarray, sampling_variance: np.ndarray) -> dict:
    """Authenticated hospital-only REML used by M3 and M4."""
    effect = np.asarray(effect, float)
    sampling_variance = np.asarray(sampling_variance, float)
    if effect.ndim != 1 or sampling_variance.shape != effect.shape:
        raise ValueError("effect and sampling_variance must be aligned vectors")

    def objective(log_tau2: float) -> float:
        variance = sampling_variance + math.exp(float(log_tau2))
        weights = 1.0 / variance
        mu = float(np.sum(weights * effect) / np.sum(weights))
        residual = effect - mu
        return 0.5 * (
            np.log(variance).sum()
            + math.log(np.sum(weights))
            + np.sum(residual * residual / variance)
            + (len(effect) - 1) * math.log(2.0 * math.pi)
        )

    opt = minimize_scalar(objective, bounds=(-27.63, 0), method="bounded", options={"xatol": 1e-10, "maxiter": 300})
    tau2 = math.exp(float(opt.x))
    variance = sampling_variance + tau2
    weights = 1.0 / variance
    mu = float(np.sum(weights * effect) / np.sum(weights))
    se = math.sqrt(1.0 / np.sum(weights))
    return {
        "converged": bool(opt.success),
        "optimizer_message": str(opt.message),
        "reml_objective": float(opt.fun),
        "mu": mu,
        "se": se,
        "sigma2_hospital": tau2,
    }


class BalancedCrossedREML:
    """Exact balanced-design algebra for the authenticated crossed REML.

    The model, REML likelihood, starts, parameter bounds, and optimizer match
    the reference crossed REML implementation.  The implementation reduces the likelihood to
    hospital-level sufficient statistics and vectorizes the known V_h block
    inverses; it does not change the estimator.
    """

    def __init__(self, covariances: list[np.ndarray]):
        cov = np.asarray(covariances, float)
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ValueError("covariances must be H x S x S")
        if not np.isfinite(cov).all():
            raise ValueError("covariances contain non-finite values")
        cov = (cov + np.swapaxes(cov, 1, 2)) / 2.0
        evals, evecs = np.linalg.eigh(cov)
        if evals.min() < -1e-10:
            raise ValueError("covariance block is not positive semidefinite")
        self.evals = np.maximum(evals, 0.0)
        self.evecs = evecs
        self.h, self.s = cov.shape[:2]
        ones = np.ones((self.h, self.s))
        self.qt_ones = np.einsum("hji,hj->hi", evecs, ones, optimize=True)
        self.eye = np.eye(self.s)

    def _quantities(self, theta: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
        vs, vh, vi = np.exp(theta)
        qty = np.einsum("hji,hj->hi", self.evecs, y, optimize=True)
        invdiag = 1.0 / (self.evals + vi)
        binv_y = np.einsum("hij,hj->hi", self.evecs, invdiag * qty, optimize=True)
        binv_1 = np.einsum("hij,hj->hi", self.evecs, invdiag * self.qt_ones, optimize=True)
        one_b_one = binv_1.sum(axis=1)
        one_b_y = binv_y.sum(axis=1)
        denom = 1.0 + vh * one_b_one
        ainv_1 = binv_1 / denom[:, None]
        ainv_y = binv_y - (vh / denom)[:, None] * binv_1 * one_b_y[:, None]

        binv = np.einsum("hik,hk,hjk->hij", self.evecs, invdiag, self.evecs, optimize=True)
        ainv = binv - (vh / denom)[:, None, None] * np.einsum("hi,hj->hij", binv_1, binv_1, optimize=True)
        summed = ainv.sum(axis=0)
        k = self.eye / vs + summed
        kf = cho_factor(k, lower=True, check_finite=False)
        gy = ainv_y.sum(axis=0)
        g1 = ainv_1.sum(axis=0)
        kigy = cho_solve(kf, gy, check_finite=False)
        kig1 = cho_solve(kf, g1, check_finite=False)

        cyy = float(np.sum(y * ainv_y) - gy @ kigy)
        c1y = float(np.sum(ainv_y) - g1 @ kigy)
        c11 = float(np.sum(ainv_1) - g1 @ kig1)
        logdet_a = float(np.log(self.evals + vi).sum() + np.log(denom).sum())
        logdet_k = float(2.0 * np.log(np.diag(kf[0])).sum())
        logdet_c = logdet_a + self.s * math.log(vs) + logdet_k
        return cyy, c1y, c11, logdet_c

    def objective(self, theta: np.ndarray, y: np.ndarray) -> float:
        try:
            cyy, c1y, c11, logdet_c = self._quantities(theta, y)
            if c11 <= 0 or not all(math.isfinite(x) for x in (cyy, c1y, c11, logdet_c)):
                return 1e100
            quad = cyy - c1y * c1y / c11
            return 0.5 * (logdet_c + math.log(c11) + quad + (self.h * self.s - 1) * math.log(2.0 * math.pi))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e100

    def fit(self, y: np.ndarray) -> dict:
        y = np.asarray(y, float)
        if y.shape != (self.h, self.s):
            raise ValueError(f"expected y shape {(self.h, self.s)}, got {y.shape}")
        starts = [np.log([1e-4, 1e-4, 1e-4]), np.log([1e-3, 1e-3, 1e-3]), np.log([1e-5, 1e-4, 1e-3])]
        fits = [minimize(self.objective, start, args=(y,), method="L-BFGS-B", bounds=[(-27.63, 0)] * 3, options={"maxiter": 300, "ftol": 1e-11}) for start in starts]
        opt = min(fits, key=lambda z: z.fun)
        _, c1y, c11, _ = self._quantities(opt.x, y)
        variance = np.exp(opt.x)
        return {
            "converged": bool(opt.success),
            "optimizer_message": str(opt.message),
            "reml_objective": float(opt.fun),
            "beta": np.array([c1y / c11]),
            "se": np.array([math.sqrt(1.0 / c11)]),
            "vcov_beta": np.array([[1.0 / c11]]),
            "sigma2_seed": float(variance[0]),
            "sigma2_hospital": float(variance[1]),
            "sigma2_interaction": float(variance[2]),
            "blup": {},
            "n_cells": self.h * self.s,
            "n_hospitals": self.h,
            "n_seeds": self.s,
        }


def variance_equivalence_count(seed: float, hospital: float, interaction: float) -> tuple[int | str, str]:
    """Frozen variance-equivalence realization count, including boundary states."""
    values = (seed, hospital, interaction)
    if any((not math.isfinite(float(v)) or float(v) < 0) for v in values):
        raise ValueError("variance components must be finite and nonnegative")
    generator_related = float(seed) + float(interaction)
    if hospital == 0 and generator_related == 0:
        return 1, "NO_TRUE_HETEROGENEITY"
    if hospital == 0:
        return "INF", "HOSPITAL_VARIANCE_ZERO_GENERATOR_VARIANCE_POSITIVE"
    if generator_related == 0:
        return 1, "NO_GENERATOR_RELATED_HETEROGENEITY"
    ratio = (Decimal(str(seed)) + Decimal(str(interaction))) / Decimal(str(hospital))
    return max(1, int(ratio.to_integral_value(rounding=ROUND_CEILING))), "FINITE_POINT_ESTIMATE"


def split_half_stable(draws: np.ndarray) -> tuple[bool, list[dict]]:
    """Apply the locked split-half comparison over all deltas and policies."""
    draws = np.asarray(draws, float)
    if len(draws) % 2:
        raise ValueError("oracle draw count must be even")
    halves = (draws[: len(draws) // 2], draws[len(draws) // 2 :])
    rows: list[dict] = []
    for delta in DELTAS:
        for policy, c in [("P95", 0.95), ("C95", 0.975)] + [("RISK_YIELD", x) for x in CONFIDENCE_GRID]:
            states = [classify_draws(h, delta, policy, c if policy == "RISK_YIELD" else None)[0] for h in halves]
            rows.append({"delta": delta, "policy": policy, "confidence": c, "half_1_state": states[0], "half_2_state": states[1], "stable": states[0] == states[1]})
    return all(r["stable"] for r in rows), rows
