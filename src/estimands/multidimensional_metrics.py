#!/usr/bin/env python3
"""Compute frozen-prediction multidimensional metrics and calibration summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BCD2 = ROOT / "outputs/postreview_upgrade/u1_bcd2_expanded_utility"
OUT = ROOT / "outputs/postreview_upgrade/u1f_multidimensional_utility"
PERF = OUT / "performance"; CAL = OUT / "calibration"; QC = OUT / "qc"


def weighted_loglik(y, eta, w, alpha, beta=1.0):
    z = alpha + beta * eta
    return float(np.sum(w * (y * (-np.logaddexp(0, -z)) + (1-y) * (-np.logaddexp(0, z)))))


def fit_offset(y, eta, w=None, max_iter=50):
    w = np.ones(len(y), float) if w is None else np.asarray(w, float)
    if np.sum(w*y) == 0 or np.sum(w*(1-y)) == 0:
        return {"status": "CALIBRATION_MODEL_NON_ESTIMABLE", "reason": "single_class", "alpha": np.nan, "se": np.nan}
    alpha = 0.0; converged = False
    for _ in range(max_iter):
        mu = expit(alpha + eta); score = np.sum(w * (y-mu)); info = np.sum(w * mu * (1-mu))
        if not np.isfinite(info) or info <= 1e-12:
            break
        step = np.clip(score / info, -5, 5); alpha += step
        if abs(step) < 1e-9:
            converged = True; break
    mu = expit(alpha + eta); info = np.sum(w * mu * (1-mu))
    se = np.sqrt(1/info) if info > 0 else np.nan
    ok = converged and np.isfinite(alpha) and np.isfinite(se) and abs(alpha) < 100
    return {"status": "ESTIMABLE" if ok else "CALIBRATION_MODEL_NON_ESTIMABLE", "reason": "" if ok else "nonconvergence_or_instability", "alpha": alpha if ok else np.nan, "se": se if ok else np.nan}


def fit_slope(y, eta, w=None, max_iter=50):
    w = np.ones(len(y), float) if w is None else np.asarray(w, float)
    sw = w.sum(); events = np.sum(w*y)
    if events == 0 or events == sw:
        return {"status": "CALIBRATION_MODEL_NON_ESTIMABLE", "reason": "single_class", "alpha": np.nan, "beta": np.nan, "alpha_se": np.nan, "beta_se": np.nan}
    theta = np.array([logit(events/sw), 0.0]); converged = False
    ll = weighted_loglik(y, eta, w, theta[0], theta[1])
    for _ in range(max_iter):
        z = theta[0] + theta[1]*eta; mu = expit(z); v = w*mu*(1-mu)
        score = np.array([np.sum(w*(y-mu)), np.sum(w*(y-mu)*eta)])
        info = np.array([[v.sum(), np.sum(v*eta)], [np.sum(v*eta), np.sum(v*eta*eta)]])
        try: step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError: break
        scale = 1.0
        while scale >= 1/1024:
            trial = theta + scale*step; trial_ll = weighted_loglik(y, eta, w, trial[0], trial[1])
            if trial_ll >= ll - 1e-8: break
            scale /= 2
        theta = trial; ll = trial_ll
        if np.max(np.abs(scale*step)) < 1e-8:
            converged = True; break
    z = theta[0]+theta[1]*eta; mu=expit(z); v=w*mu*(1-mu)
    info=np.array([[v.sum(),np.sum(v*eta)],[np.sum(v*eta),np.sum(v*eta*eta)]])
    try:
        cov=np.linalg.inv(info); diag=np.diag(cov); ses=np.sqrt(diag) if np.all(diag>0) else np.array([np.nan,np.nan]); cond=np.linalg.cond(info)
    except np.linalg.LinAlgError:
        ses=np.array([np.nan,np.nan]); cond=np.inf
    ok=converged and np.isfinite(theta).all() and np.isfinite(ses).all() and np.max(np.abs(theta))<100 and cond<1e12
    return {"status":"ESTIMABLE" if ok else "CALIBRATION_MODEL_NON_ESTIMABLE", "reason":"" if ok else "nonconvergence_separation_or_instability", "alpha":theta[0] if ok else np.nan, "beta":theta[1] if ok else np.nan, "alpha_se":ses[0] if ok else np.nan, "beta_se":ses[1] if ok else np.nan}


def calibration_bins(y, p, requested=10):
    if np.unique(p).size == 1:
        codes=np.zeros(len(p), dtype=int)
    else:
        try:
            codes=pd.qcut(p, q=requested, labels=False, duplicates="drop").to_numpy() if isinstance(pd.qcut(p, q=requested, labels=False, duplicates="drop"), pd.Series) else np.asarray(pd.qcut(p, q=requested, labels=False, duplicates="drop"))
        except ValueError:
            codes=np.zeros(len(p), dtype=int)
    uniq=np.unique(codes[~pd.isna(codes)]); remap={v:i+1 for i,v in enumerate(uniq)}; codes=np.array([remap[v] for v in codes], int)
    rows=[]
    for b in sorted(np.unique(codes)):
        m=codes==b
        rows.append({"bin":int(b),"N":int(m.sum()),"events":int(y[m].sum()),"mean_predicted_probability":float(p[m].mean()),"observed_event_proportion":float(y[m].mean()),"requested_bins":requested,"actual_bins":len(uniq)})
    ece=sum(r["N"]/len(y)*abs(r["mean_predicted_probability"]-r["observed_event_proportion"]) for r in rows)
    return rows, float(ece)


def metrics(y, p):
    n=len(y); events=int(y.sum()); pi=events/n
    auroc=roc_auc_score(y,p); ap=average_precision_score(y,p); aps=(ap-pi)/(1-pi)
    bs=float(np.mean((p-y)**2)); bs0=pi*(1-pi); bss=1-bs/bs0
    pc=np.clip(p,1e-15,1-1e-15); ll=float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))); ll0=float(-(pi*np.log(pi)+(1-pi)*np.log(1-pi))); lls=1-ll/ll0
    eta=logit(np.clip(p,1e-6,1-1e-6)); oi=fit_offset(y,eta); sl=fit_slope(y,eta); bins,ece=calibration_bins(y,p)
    status="ESTIMABLE" if oi["status"]=="ESTIMABLE" and sl["status"]=="ESTIMABLE" else "CALIBRATION_MODEL_NON_ESTIMABLE"
    return {"N":n,"events":events,"prevalence":pi,"AUROC":auroc,"AUPRC":ap,"AP_skill":aps,"Brier":bs,"Brier_skill":bss,"log_loss":ll,"log_loss_skill":lls,
            "calibration_intercept":oi["alpha"],"calibration_intercept_SE":oi["se"],"calibration_intercept_CI_lower":oi["alpha"]-1.96*oi["se"] if oi["status"]=="ESTIMABLE" else np.nan,"calibration_intercept_CI_upper":oi["alpha"]+1.96*oi["se"] if oi["status"]=="ESTIMABLE" else np.nan,
            "calibration_model_intercept":sl["alpha"],"calibration_model_intercept_SE":sl["alpha_se"],"calibration_slope":sl["beta"],"calibration_slope_SE":sl["beta_se"],"calibration_slope_CI_lower":sl["beta"]-1.96*sl["beta_se"] if sl["status"]=="ESTIMABLE" else np.nan,"calibration_slope_CI_upper":sl["beta"]+1.96*sl["beta_se"] if sl["status"]=="ESTIMABLE" else np.nan,
            "abs_calibration_intercept":abs(oi["alpha"]) if oi["status"]=="ESTIMABLE" else np.nan,"abs_calibration_slope_deviation":abs(sl["beta"]-1) if sl["status"]=="ESTIMABLE" else np.nan,"ECE":ece,"calibration_status":status,"calibration_failure_reason":";".join(filter(None,[oi["reason"],sl["reason"]]))}, bins


def main():
    if not json.loads((QC/"U1F_PREEXECUTION_GATE.json").read_text())["preexecution_gate_pass"]: raise SystemExit("preexecution failed")
    pr=pd.read_csv(BCD2/"manifests/prediction_registry.csv"); rows=[]; bin_rows=[]; trace=[]
    for i,r in enumerate(pr.itertuples(index=False),1):
        d=pd.read_parquet(ROOT/r.prediction_artifact,columns=["y_true","predicted_probability"]); y=d.y_true.to_numpy(np.int8); p=d.predicted_probability.to_numpy(float)
        m,bins=metrics(y,p); training_type="REAL_TRAINED" if r.generator=="REAL_REFERENCE" else "SYNTHETIC_TRAINED"
        row={"generator":r.generator,"seed":r.seed,"model":r.model,"domain":r.domain,"training_type":training_type,**m,"utility_estimability":"UTILITY_ESTIMABLE","prediction_artifact":r.prediction_artifact,"prediction_sha256":r.prediction_sha256,"source":r.source}
        rows.append(row)
        for q in bins: bin_rows.append({"generator":r.generator,"seed":r.seed,"model":r.model,"domain":r.domain,"training_type":training_type,**q})
        trace.append({"generator":r.generator,"seed":r.seed,"model":r.model,"domain":r.domain,"prediction_artifact":r.prediction_artifact,"prediction_sha256":r.prediction_sha256,"label_source":"frozen_prediction_artifact:y_true","probability_source":"frozen_prediction_artifact:predicted_probability","metric_script":"02_point_metrics.py","traceability_complete":True})
        if i%25==0: print(f"point_cells={i}/{len(pr)}",flush=True)
    perf=pd.DataFrame(rows)
    bcdperf=pd.read_csv(BCD2/"manifests/prediction_registry.csv")[["generator","seed","model","domain","auroc"]]
    rec=perf.merge(bcdperf,on=["generator","seed","model","domain"]); rec["abs_error"]=abs(rec.AUROC-rec.auroc)
    rec.to_csv(QC/"auroc_reconciliation.csv",index=False)
    if rec.abs_error.max()>1e-12: raise RuntimeError("AUROC reconciliation failed")
    syn=perf.loc[perf.training_type.eq("SYNTHETIC_TRAINED")].copy(); real=perf.loc[perf.training_type.eq("REAL_TRAINED")].copy()
    complete=pd.read_csv(BCD2/"performance/u1_bcd2_complete_performance_grid.csv")
    idcols=["generator","seed","model","domain"]
    metriccols=[c for c in syn.columns if c not in idcols and c not in ["utility_estimability"]]
    grid=complete[idcols+["utility_estimability"]].merge(syn[idcols+metriccols],on=idcols,how="left")
    grid.to_csv(PERF/"u1f_complete_performance_grid.csv",index=False); real.to_csv(PERF/"u1f_real_reference_performance.csv",index=False)
    calcols=["generator","seed","model","domain","training_type","N","events","prevalence","calibration_intercept","calibration_intercept_SE","calibration_intercept_CI_lower","calibration_intercept_CI_upper","calibration_model_intercept","calibration_model_intercept_SE","calibration_slope","calibration_slope_SE","calibration_slope_CI_lower","calibration_slope_CI_upper","abs_calibration_intercept","abs_calibration_slope_deviation","ECE","calibration_status","calibration_failure_reason"]
    perf[calcols].to_csv(CAL/"u1f_calibration_summary.csv",index=False); pd.DataFrame(bin_rows).to_csv(CAL/"u1f_calibration_curve_bins.csv",index=False)
    pd.DataFrame(trace).to_csv(QC/"U1F_NUMERICAL_TRACEABILITY.csv",index=False)
    summary={"performance_cells":len(perf),"synthetic_estimable":len(syn),"real_reference":len(real),"complete_grid_rows":len(grid),"collapsed_rows":int(grid.utility_estimability.ne("UTILITY_ESTIMABLE").sum()),"auroc_reconciliation_failures":int((rec.abs_error>1e-12).sum()),"auroc_reconciliation_max_abs_error":float(rec.abs_error.max()),"calibration_estimable":int(perf.calibration_status.eq("ESTIMABLE").sum()),"calibration_nonestimable":int(perf.calibration_status.ne("ESTIMABLE").sum()),"calibration_bin_rows":len(bin_rows),"traceability_complete":bool(pd.DataFrame(trace).traceability_complete.all())}
    (QC/"U1F_POINT_METRICS_QC.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
