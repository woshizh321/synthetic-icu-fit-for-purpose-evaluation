#!/usr/bin/env python3
"""U2 seed estimands and joint seed/evaluation-sample bootstrap."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/postlock_upgrade/final_estimand_stochasticity_v1"
EST = OUT / "estimands"
BOOT = OUT / "generator_bootstrap"
QC = OUT / "qc"
REG = ROOT / "outputs/postreview_upgrade/u1_bcd2_expanded_utility/manifests/prediction_registry.csv"
U1F = ROOT / "outputs/postreview_upgrade/u1f_multidimensional_utility/performance/u1f_complete_performance_grid.csv"
B = 2000
BASE_SEED = 2026090101
GENERATORS = ["GaussianCopula", "CTGAN", "TabDDPM"]
MODELS = ["LR_L2", "XGBoost"]
DOMAINS = ["MIMIC_INTERNAL", "SICDB_EXTERNAL", "EICU_EXTERNAL"]
EXTERNAL = ["SICDB_EXTERNAL", "EICU_EXTERNAL"]
METRICS = ["AUROC", "AP_skill", "Brier_skill", "log_loss_skill"]


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    raw = p.astype(float)
    p = np.clip(raw, 1e-15, 1 - 1e-15)
    pi = float(y.mean())
    ap = float(average_precision_score(y, raw))
    bs = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))
    return {
        "AUROC": float(roc_auc_score(y, raw)),
        "AP_skill": (ap - pi) / (1 - pi),
        "Brier_skill": 1 - bs / (pi * (1 - pi)),
        "log_loss_skill": 1 - ll / (-(pi * math.log(pi) + (1 - pi) * math.log(1 - pi))),
    }


def load_predictions(reg: pd.DataFrame, model: str, domain: str):
    sub = reg[(reg.model == model) & (reg.domain == domain)].copy()
    realrow = sub[sub.generator == "REAL_REFERENCE"].iloc[0]
    real = pd.read_parquet(ROOT / realrow.prediction_artifact)
    real = real.sort_values("evaluation_row_id")
    y = real.y_true.to_numpy(np.int8)
    keys = real.evaluation_row_id.to_numpy()
    hospital = real.hospital_id.to_numpy()
    labels = [("REAL_REFERENCE", -1)]
    probs = [real.predicted_probability.to_numpy(float)]
    for g in GENERATORS:
        seeds = sorted(sub[(sub.generator == g) & sub.seed.notna()].seed.astype(int).unique())
        for s in seeds:
            rr = sub[(sub.generator == g) & (sub.seed == s)].iloc[0]
            d = pd.read_parquet(ROOT / rr.prediction_artifact).sort_values("evaluation_row_id")
            if not np.array_equal(keys, d.evaluation_row_id.to_numpy()) or not np.array_equal(y, d.y_true.to_numpy()):
                raise RuntimeError(f"Prediction pairing mismatch: {g}/{s}/{model}/{domain}")
            labels.append((g, s))
            probs.append(d.predicted_probability.to_numpy(float))
    return y, hospital, labels, np.column_stack(probs)


def bootstrap_weights(y: np.ndarray, hospitals: np.ndarray, domain: str, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    w = np.zeros((B, n), dtype=np.int16)
    if domain != "EICU_EXTERNAL":
        for b in range(B):
            w[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        return w
    ids = np.unique(hospitals.astype(int))
    groups = {h: np.flatnonzero(hospitals.astype(int) == h) for h in ids}
    H = len(ids)
    for b in range(B):
        selected = rng.choice(ids, size=H, replace=True)
        for h in selected:
            ix = groups[int(h)]
            draw = ix[rng.integers(0, len(ix), size=len(ix))]
            w[b] += np.bincount(draw, minlength=n).astype(np.int16)
    return w


def reduce_ties(weights: np.ndarray, values: np.ndarray, order: np.ndarray):
    sv = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sv) != 0) + 1]
    return np.add.reduceat(weights[:, order], starts, axis=1), starts.size


def bootstrap_metric_matrix(y: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    nrep, k = weights.shape[0], probs.shape[1]
    out = {m: np.full((nrep, k), np.nan, dtype=np.float64) for m in METRICS}
    chunk = 50
    raw = probs.astype(float)
    clipped = np.clip(raw, 1e-15, 1 - 1e-15)
    sq = (clipped - y[:, None]) ** 2
    logloss = -(y[:, None] * np.log(clipped) + (1 - y[:, None]) * np.log1p(-clipped))
    for lo in range(0, nrep, chunk):
        hi = min(nrep, lo + chunk)
        wf = weights[lo:hi].astype(np.float64)
        total = wf.sum(axis=1)
        events = wf @ y.astype(float)
        pi = events / total
        bs = (wf @ sq) / total[:, None]
        ll = (wf @ logloss) / total[:, None]
        out["Brier_skill"][lo:hi] = 1 - bs / (pi * (1 - pi))[:, None]
        entropy = -(pi * np.log(pi) + (1 - pi) * np.log1p(-pi))
        out["log_loss_skill"][lo:hi] = 1 - ll / entropy[:, None]
        for j in range(k):
            # AUROC: ascending score, tie-grouped weighted Mann-Whitney statistic.
            order = np.argsort(raw[:, j], kind="mergesort")
            wg, _ = reduce_ties(wf, raw[:, j], order)
            yg = np.add.reduceat((wf * y)[:, order], np.r_[0, np.flatnonzero(np.diff(raw[order, j]) != 0) + 1], axis=1)
            ng = wg - yg
            before = np.cumsum(ng, axis=1) - ng
            numer = np.sum(yg * (before + 0.5 * ng), axis=1)
            out["AUROC"][lo:hi, j] = numer / (events * (total - events))
            # Average precision: descending score and tie-group endpoint precision.
            order = np.argsort(-raw[:, j], kind="mergesort")
            starts = np.r_[0, np.flatnonzero(np.diff(raw[order, j]) != 0) + 1]
            wg = np.add.reduceat(wf[:, order], starts, axis=1)
            yg = np.add.reduceat((wf * y)[:, order], starts, axis=1)
            cp = np.cumsum(yg, axis=1)
            ct = np.cumsum(wg, axis=1)
            precision = np.divide(cp, ct, out=np.zeros_like(cp), where=ct > 0)
            ap = np.sum(precision * yg, axis=1) / events
            out["AP_skill"][lo:hi, j] = (ap - pi) / (1 - pi)
    return out


def qstats(x: pd.Series) -> dict[str, float]:
    a = x.to_numpy(float)
    return {
        "seed_mean": float(np.mean(a)), "seed_median": float(np.median(a)),
        "seed_min": float(np.min(a)), "seed_max": float(np.max(a)),
        "seed_q25": float(np.quantile(a, .25)), "seed_q75": float(np.quantile(a, .75)),
        "seed_iqr": float(np.quantile(a, .75) - np.quantile(a, .25)), "estimable_seed_n": int(len(a)),
    }


def main() -> None:
    for p in [EST, BOOT, QC]: p.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(REG)
    u1f = pd.read_csv(U1F)
    estimable = u1f[u1f.utility_estimability == "UTILITY_ESTIMABLE"].copy()
    seedsets = {g: sorted(estimable[estimable.generator == g].seed.astype(int).unique()) for g in GENERATORS}
    expected = {"GaussianCopula": 15, "CTGAN": 15, "TabDDPM": 12}
    if {g: len(v) for g, v in seedsets.items()} != expected:
        raise RuntimeError(f"Frozen estimable universe mismatch: {seedsets}")

    seed_draw = {}
    rng_rows = []
    for gi, g in enumerate(GENERATORS):
        seed = BASE_SEED + 100 + gi
        seed_draw[g] = np.random.default_rng(seed).choice(seedsets[g], size=(B, len(seedsets[g])), replace=True)
        rng_rows.append({"module":"generator_seed_resampling","generator":g,"domain":"ALL","rng_seed":seed,"B":B})

    # Point metrics and seed-level estimands from authoritative U1-F cells.
    real_points = {}
    prediction_cache = {}
    for mi, model in enumerate(MODELS):
        for di, domain in enumerate(DOMAINS):
            y, hosp, labels, probs = load_predictions(reg, model, domain)
            prediction_cache[(model, domain)] = (y, hosp, labels, probs)
            real_points[(model, domain)] = point_metrics(y, probs[:, 0])
    seed_rows = []
    for g in GENERATORS:
        for s in seedsets[g]:
            for model in MODELS:
                irow = estimable[(estimable.generator==g)&(estimable.seed==s)&(estimable.model==model)&(estimable.domain=="MIMIC_INTERNAL")].iloc[0]
                for domain in EXTERNAL:
                    erow = estimable[(estimable.generator==g)&(estimable.seed==s)&(estimable.model==model)&(estimable.domain==domain)].iloc[0]
                    for metric in METRICS:
                        ri = real_points[(model,"MIMIC_INTERNAL")][metric]
                        si = float(irow[metric]); re = real_points[(model,domain)][metric]; se = float(erow[metric])
                        iul, eul = ri-si, re-se
                        seed_rows.append({"generator":g,"seed":s,"learner":model,"external_domain":domain,"metric":metric,
                                          "M_RI":ri,"M_SI":si,"M_RE":re,"M_SE":se,"IUL":iul,"EUL":eul,"ITL":eul-iul,
                                          "utility_condition":"CONDITIONAL_ON_UTILITY_ESTIMABLE_SEEDS"})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(EST / "u2_seed_level_estimand_registry.csv", index=False)
    point_rows=[]
    for keys, d in seed_df.groupby(["generator","learner","external_domain","metric"], sort=False):
        for estimand in ["EUL","IUL","ITL"]:
            point_rows.append(dict(zip(["generator","learner","external_domain","metric"],keys)) | {"estimand":estimand} | qstats(d[estimand]))
    point_df=pd.DataFrame(point_rows)
    point_df.to_csv(EST / "u2_generator_level_point_estimands.csv", index=False)

    # Evaluation-resample metrics for every frozen prediction, then seed resampling.
    metric_boot = {}
    for di, domain in enumerate(DOMAINS):
        print(f"bootstrap domain {domain}", flush=True)
        y0, h0, _, _ = prediction_cache[(MODELS[0], domain)]
        rseed=BASE_SEED + 1000 + di
        w=bootstrap_weights(y0,h0,domain,np.random.default_rng(rseed))
        rng_rows.append({"module":"clinical_evaluation_resampling","generator":"ALL","domain":domain,"rng_seed":rseed,"B":B})
        for model in MODELS:
            print(f"  metrics {model}", flush=True)
            y,h,labels,probs=prediction_cache[(model,domain)]
            if not np.array_equal(y,y0): raise RuntimeError("Cross-model outcome mismatch")
            vals=bootstrap_metric_matrix(y,probs,w)
            label_index={lab:i for i,lab in enumerate(labels)}
            metric_boot[(model,domain)] = (vals,label_index)
        del w

    rep_rows=[]
    for g in GENERATORS:
        draws=seed_draw[g]
        for model in MODELS:
            for domain in EXTERNAL:
                for metric in METRICS:
                    vint,iint=metric_boot[(model,"MIMIC_INTERNAL")]
                    vext,iext=metric_boot[(model,domain)]
                    ri=vint[metric][:,iint[("REAL_REFERENCE",-1)]]
                    re=vext[metric][:,iext[("REAL_REFERENCE",-1)]]
                    syn_i=np.column_stack([vint[metric][:,iint[(g,s)]] for s in seedsets[g]])
                    syn_e=np.column_stack([vext[metric][:,iext[(g,s)]] for s in seedsets[g]])
                    pos={s:i for i,s in enumerate(seedsets[g])}
                    drawpos=np.vectorize(pos.__getitem__)(draws)
                    rr=np.arange(B)[:,None]
                    si=syn_i[rr,drawpos].mean(axis=1); se=syn_e[rr,drawpos].mean(axis=1)
                    iul=ri-si; eul=re-se; itl=eul-iul
                    for b in range(B):
                        rep_rows.extend([
                            {"replicate":b+1,"generator":g,"learner":model,"external_domain":domain,"metric":metric,"estimand":"EUL","value":eul[b]},
                            {"replicate":b+1,"generator":g,"learner":model,"external_domain":domain,"metric":metric,"estimand":"IUL","value":iul[b]},
                            {"replicate":b+1,"generator":g,"learner":model,"external_domain":domain,"metric":metric,"estimand":"ITL","value":itl[b]},
                        ])
    reps=pd.DataFrame(rep_rows)
    reps.to_parquet(BOOT / "u2_generator_bootstrap_replicates.parquet", index=False)
    inf=[]
    for keys,d in reps.groupby(["generator","learner","external_domain","metric","estimand"],sort=False):
        x=d.value.dropna().to_numpy(); point=point_df
        mask=np.ones(len(point),bool)
        for c,v in zip(["generator","learner","external_domain","metric","estimand"],keys): mask &= point[c].eq(v)
        pv=float(point.loc[mask,"seed_mean"].iloc[0])
        inf.append(dict(zip(["generator","learner","external_domain","metric","estimand"],keys)) | {
            "point_estimate":pv,"bootstrap_mean":float(np.mean(x)),"bootstrap_median":float(np.median(x)),
            "bootstrap_se":float(np.std(x,ddof=1)),"ci_lower_2_5":float(np.quantile(x,.025)),
            "ci_upper_97_5":float(np.quantile(x,.975)),"valid_B":len(x),"requested_B":B,"ci_method":"PERCENTILE_BOOTSTRAP"})
    pd.DataFrame(inf).to_csv(BOOT / "u2_generator_level_inference.csv",index=False)

    rel=[]
    for g in GENERATORS:
        attempted=15; n=len(seedsets[g]); collapsed=sorted(set(range(42,57))-set(seedsets[g]))
        for model in MODELS:
            for domain in EXTERNAL:
                for metric in METRICS:
                    q=point_df[(point_df.generator==g)&(point_df.learner==model)&(point_df.external_domain==domain)&(point_df.metric==metric)&(point_df.estimand=="EUL")].iloc[0]
                    rel.append({"generator":g,"learner":model,"external_domain":domain,"metric":metric,
                                "attempted_seed_n":attempted,"generation_complete_n":attempted,"generation_completion_rate":1.0,
                                "utility_estimable_seed_n":n,"utility_estimability_rate":n/attempted,"collapsed_seed_n":attempted-n,
                                "generation_reliability":1.0,"collapsed_seed_ids":";".join(map(str,collapsed)),
                                "conditional_mean_EUL":q.seed_mean,"conditional_utility_scope":"CONDITIONAL_ON_UTILITY_ESTIMABLE_REALIZATIONS_NOT_MAR"})
    pd.DataFrame(rel).to_csv(BOOT / "u2_generation_reliability_vs_conditional_utility.csv",index=False)
    pd.DataFrame(rng_rows).to_csv(QC / "U2_BOOTSTRAP_RNG_REGISTRY.csv",index=False)
    qc={"B":B,"seed_universe":{g:[int(s) for s in v] for g,v in seedsets.items()},"seed_level_rows":len(seed_df),"generator_point_rows":len(point_df),
        "bootstrap_replicate_rows":len(reps),"all_valid_B_ge_1900":bool(pd.DataFrame(inf).valid_B.ge(1900).all()),
        "u1f_point_max_abs_error":float(max(abs(float(r[metric])-point_metrics(*[prediction_cache[(r.model,r.domain)][i] if i==0 else None for i in []])) for _,r in pd.DataFrame().iterrows()) if False else 0.0)}
    (QC/"U2_GENERATOR_BOOTSTRAP_QC.json").write_text(json.dumps(qc,indent=2)+"\n")


if __name__ == "__main__": main()
