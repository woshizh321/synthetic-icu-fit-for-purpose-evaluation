#!/usr/bin/env python3
"""U2 eICU paired covariance blocks, crossed REML models, and D_W models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/postlock_upgrade/final_estimand_stochasticity_v1"
HC = OUT / "hospital_crossed"
DD = OUT / "domain_distance"
QC = OUT / "qc"
REG = ROOT / "outputs/postreview_upgrade/u1_bcd2_expanded_utility/manifests/prediction_registry.csv"
PERF = ROOT / "outputs/postreview_upgrade/u1_bcd2_expanded_utility/performance/u1_bcd2_complete_performance_grid.csv"
ELIG = ROOT / "outputs/postreview_upgrade/u1h_b12_protocol_conformant_repair_r2/hospital_effects/hospital_eligibility_registry_repair.csv"
DW = ROOT / "outputs/postreview_upgrade/u1h_b12_protocol_conformant_repair_r1/domain_distance/u1h_b12_r1_DW_all_hospitals.csv"
B = 2000
BASE_SEED = 2027091001
GENERATORS = ["GaussianCopula", "CTGAN", "TabDDPM"]
MODELS = ["LR_L2", "XGBoost"]
THRESHOLDS = {"100_10_10":"eligible_100_10_10", "200_20_20":"eligible_200_20_20", "300_30_30":"eligible_300_30_30"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model: str):
    reg=pd.read_csv(REG)
    sub=reg[(reg.model==model)&(reg.domain=="EICU_EXTERNAL")]
    realrow=sub[sub.generator=="REAL_REFERENCE"].iloc[0]
    real=pd.read_parquet(ROOT/realrow.prediction_artifact).sort_values("evaluation_row_id")
    keys=real.evaluation_row_id.to_numpy(); y=real.y_true.to_numpy(np.int8); h=real.hospital_id.to_numpy(int)
    labels=[("REAL_REFERENCE",-1)]; probs=[real.predicted_probability.to_numpy(float)]
    for g in GENERATORS:
        for s in sorted(sub[(sub.generator==g)&sub.seed.notna()].seed.astype(int).unique()):
            rr=sub[(sub.generator==g)&(sub.seed==s)].iloc[0]
            d=pd.read_parquet(ROOT/rr.prediction_artifact).sort_values("evaluation_row_id")
            if not np.array_equal(keys,d.evaluation_row_id.to_numpy()) or not np.array_equal(y,d.y_true.to_numpy()):
                raise RuntimeError(f"Pairing failure {g}/{s}/{model}")
            labels.append((g,s)); probs.append(d.predicted_probability.to_numpy(float))
    return y,h,labels,np.column_stack(probs)


def auc_bootstrap(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n=len(y); w=rng.multinomial(n,np.full(n,1/n),size=B).astype(np.int16)
    total=w.sum(axis=1).astype(float); events=w@y.astype(float)
    ans=np.full((B,p.shape[1]),np.nan)
    for j in range(p.shape[1]):
        order=np.argsort(p[:,j],kind="mergesort"); sv=p[order,j]
        starts=np.r_[0,np.flatnonzero(np.diff(sv)!=0)+1]
        for lo in range(0,B,100):
            hi=min(B,lo+100); wf=w[lo:hi].astype(float)
            wg=np.add.reduceat(wf[:,order],starts,axis=1)
            yg=np.add.reduceat((wf*y)[:,order],starts,axis=1); ng=wg-yg
            before=np.cumsum(ng,axis=1)-ng
            den=events[lo:hi]*(total[lo:hi]-events[lo:hi])
            ans[lo:hi,j]=np.divide(np.sum(yg*(before+.5*ng),axis=1),den,out=np.full(hi-lo,np.nan),where=den>0)
    return ans


def psd_cov(x: np.ndarray):
    v=np.cov(x,rowvar=False,ddof=1); v=(v+v.T)/2
    ev,q=np.linalg.eigh(v); tol=1e-10*max(1.0,float(np.max(np.abs(ev))))
    if float(ev.min()) < -tol: raise RuntimeError(f"PSD failure min={ev.min()} tol={tol}")
    corrected=bool(np.any(ev<0)); ev2=np.maximum(ev,0); vv=(q*ev2)@q.T; vv=(vv+vv.T)/2
    return vv,float(ev.min()),float(np.linalg.eigvalsh(vv).min()),tol,corrected


def fit_reml(cells: pd.DataFrame, covs: dict[int,np.ndarray], dw: bool=False, all_seeds=None):
    cells=cells.sort_values(["hospital_id","seed"]).reset_index(drop=True)
    hospitals=sorted(cells.hospital_id.unique()); seeds=sorted(cells.seed.unique()); smap={s:i for i,s in enumerate(seeds)}
    y=cells.EUL.to_numpy(float); X=np.column_stack([np.ones(len(cells)),cells.Z_DW.to_numpy(float)]) if dw else np.ones((len(cells),1))
    blocks=[]; offset=0
    for h in hospitals:
        d=cells[cells.hospital_id==h]; ss=d.seed.astype(int).tolist(); full=covs[int(h)]
        allseeds=sorted(all_seeds if all_seeds is not None else seeds)
        ix=[allseeds.index(s) for s in ss]; V=full[np.ix_(ix,ix)]
        z=np.zeros((len(ss),len(seeds))); z[np.arange(len(ss)),[smap[s] for s in ss]]=1
        blocks.append((int(h),slice(offset,offset+len(ss)),V,z)); offset+=len(ss)

    def solve(theta, Q, details=False):
        vs,vh,vi=np.exp(theta)
        AinvQ=np.zeros_like(Q); AinvZ=np.zeros((len(y),len(seeds))); logdet=0.0
        for _,sl,V,z in blocks:
            A=V+vi*np.eye(len(V))+vh*np.ones_like(V)
            cf=cho_factor(A,lower=True,check_finite=False); logdet+=2*np.log(np.diag(cf[0])).sum()
            AinvQ[sl]=cho_solve(cf,Q[sl],check_finite=False); AinvZ[sl]=cho_solve(cf,z,check_finite=False)
        if vs>0:
            K=np.eye(len(seeds))/vs + sum((b[3].T@AinvZ[b[1]] for b in blocks),np.zeros((len(seeds),len(seeds))))
            kf=cho_factor(K,lower=True,check_finite=False)
            ztaq=sum((b[3].T@AinvQ[b[1]] for b in blocks),np.zeros((len(seeds),Q.shape[1])))
            CinvQ=AinvQ-AinvZ@cho_solve(kf,ztaq,check_finite=False)
            logdet += len(seeds)*math.log(vs)+2*np.log(np.diag(kf[0])).sum()
        else: CinvQ=AinvQ
        return (CinvQ,logdet,AinvZ) if details else (CinvQ,logdet)

    def obj(theta):
        Q=np.column_stack([y,X]); ci,ld=solve(theta,Q)
        ciy=ci[:,0]; ciX=ci[:,1:]; xtx=X.T@ciX
        try: beta=np.linalg.solve(xtx,X.T@ciy)
        except np.linalg.LinAlgError: return 1e100
        resid=y-X@beta; cir,_=solve(theta,resid[:,None]); quad=float(resid@cir[:,0])
        sign,ldx=np.linalg.slogdet(xtx)
        if sign<=0 or not np.isfinite(quad): return 1e100
        return .5*(ld+ldx+quad+(len(y)-X.shape[1])*math.log(2*math.pi))

    starts=[np.log([1e-4,1e-4,1e-4]),np.log([1e-3,1e-3,1e-3]),np.log([1e-5,1e-4,1e-3])]
    fits=[minimize(obj,s,method="L-BFGS-B",bounds=[(-27.63,0)]*3,options={"maxiter":300,"ftol":1e-11}) for s in starts]
    opt=min(fits,key=lambda z:z.fun)
    ci,_=solve(opt.x,np.column_stack([y,X])); ciy=ci[:,0]; ciX=ci[:,1:]
    vc=np.linalg.inv(X.T@ciX); beta=vc@(X.T@ciy); se=np.sqrt(np.diag(vc)); vars_=np.exp(opt.x)
    resid=y-X@beta; cir,_=solve(opt.x,resid[:,None]); cir=cir[:,0]
    blup={}
    for h,sl,_,_ in blocks: blup[h]=float(vars_[1]*cir[sl].sum())
    return {"converged":bool(opt.success),"optimizer_message":str(opt.message),"reml_objective":float(opt.fun),
            "beta":beta,"se":se,"vcov_beta":vc,"sigma2_seed":vars_[0],"sigma2_hospital":vars_[1],"sigma2_interaction":vars_[2],"blup":blup,
            "n_cells":len(y),"n_hospitals":len(hospitals),"n_seeds":len(seeds)}


def result_row(g,m,t,label,fit):
    mu=float(fit["beta"][0]); sem=float(fit["se"][0]); vs,vh,vi=[float(fit[k]) for k in ["sigma2_seed","sigma2_hospital","sigma2_interaction"]]
    total=vs+vh+vi
    return {"generator":g,"learner":m,"threshold":t,"seed_scope":label,"hospital_n":fit["n_hospitals"],"seed_n":fit["n_seeds"],"cell_n":fit["n_cells"],
            "mu_EUL":mu,"mu_SE":sem,"mu_CI_lower":mu-1.96*sem,"mu_CI_upper":mu+1.96*sem,
            "sigma2_seed":vs,"sigma2_hospital":vh,"sigma2_seed_x_hospital":vi,
            "tau_seed":math.sqrt(vs),"tau_hospital":math.sqrt(vh),"tau_seed_x_hospital":math.sqrt(vi),
            "share_seed":vs/total if total>0 else np.nan,"share_hospital":vh/total if total>0 else np.nan,"share_interaction":vi/total if total>0 else np.nan,
            "hospital_average_seed_PI95_lower":mu-1.96*math.sqrt(sem**2+vh),"hospital_average_seed_PI95_upper":mu+1.96*math.sqrt(sem**2+vh),
            "new_seed_new_hospital_PI95_lower":mu-1.96*math.sqrt(sem**2+vs+vh+vi),"new_seed_new_hospital_PI95_upper":mu+1.96*math.sqrt(sem**2+vs+vh+vi),
            "method":"CUSTOM_EXACT_GAUSSIAN_REML_KNOWN_BLOCK_V","converged":fit["converged"],"optimizer_message":fit["optimizer_message"]}


def main():
    for p in [HC,DD,QC]: p.mkdir(parents=True,exist_ok=True)
    elig=pd.read_csv(ELIG); dw=pd.read_csv(DW)
    if len(dw)!=208 or sha(DW)!="4e42a318288d00cc2e040f1b1d9c4b9058d64ac76e64b56611804ea62e1657f9": raise RuntimeError("D_W authority failure")
    mean=float(dw.D_W.mean()); sd=float(dw.D_W.std(ddof=1)); zdw=dw[["hospital_id","D_W"]].copy(); zdw["Z_DW"]=(zdw.D_W-mean)/sd
    zpath=DD/"u2_DW_standardized_all_hospitals.csv"; zdw.to_csv(zpath,index=False)
    lock={"source":str(DW.relative_to(ROOT)),"source_sha256":sha(DW),"hospital_n":208,"mean_DW":mean,"sample_sd_DW_ddof1":sd,"standardized_file":str(zpath.relative_to(ROOT)),"standardized_file_sha256":sha(zpath),"outcome_accessed_before_lock":False}
    (DD/"u2_DW_standardization_lock.json").write_text(json.dumps(lock,indent=2)+"\n")
    if {k:int(elig[v].sum()) for k,v in THRESHOLDS.items()}!={"100_10_10":127,"200_20_20":85,"300_30_30":66}: raise RuntimeError("Eligibility count mismatch")
    perf=pd.read_csv(PERF); est=perf[perf.utility_estimability=="UTILITY_ESTIMABLE"]
    seedsets={g:sorted(est[est.generator==g].seed.astype(int).unique()) for g in GENERATORS}
    adequate={m:sorted(est[(est.generator=="TabDDPM")&(est.model==m)&(est.domain=="MIMIC_INTERNAL")&(est.auroc>=.55)].seed.astype(int).unique()) for m in MODELS}
    eligible_h=sorted(elig.loc[elig.eligible_100_10_10,"hospital_id"].astype(int))
    cell_rows=[]; cov_rows=[]; manifests=[]; covdict={}
    for mi,m in enumerate(MODELS):
        print(f"hospital covariance {m}",flush=True)
        y,h,labels,probs=load_model(m); li={lab:i for i,lab in enumerate(labels)}
        for hi,hid in enumerate(eligible_h):
            ix=np.flatnonzero(h==hid); yy=y[ix]; pp=probs[ix]
            aucpoint={lab:roc_auc_score(yy,pp[:,j]) for j,lab in enumerate(labels)}
            ab=auc_bootstrap(yy,pp,np.random.default_rng(BASE_SEED+mi*1000+hi))
            for g in GENERATORS:
                seeds=seedsets[g]; cols=[li[(g,s)] for s in seeds]; eulb=ab[:,[li[("REAL_REFERENCE",-1)]]]-ab[:,cols]
                valid=np.isfinite(eulb).all(axis=1); vv,mineig,post,tol,corr=psd_cov(eulb[valid])
                covdict[(g,m,int(hid))]=vv
                manifests.append({"hospital_id":hid,"generator":g,"learner":m,"seed_n":len(seeds),"requested_B":B,"valid_B":int(valid.sum()),"ddof":1,"min_eigen_before":mineig,"min_eigen_after":post,"psd_tolerance":tol,"spectral_correction":corr,"status":"PASS" if valid.sum()>=1000 else "FAIL_VALID_B"})
                for a,s1 in enumerate(seeds):
                    for b,s2 in enumerate(seeds): cov_rows.append({"hospital_id":hid,"generator":g,"learner":m,"row_seed":s1,"column_seed":s2,"covariance":vv[a,b]})
                for s in seeds:
                    cell_rows.append({"hospital_id":hid,"generator":g,"learner":m,"seed":s,"N":len(ix),"events":int(yy.sum()),"non_events":int(len(ix)-yy.sum()),
                                      "AUC_RE_h":aucpoint[("REAL_REFERENCE",-1)],"AUC_SE_sh":aucpoint[(g,s)],"EUL":aucpoint[("REAL_REFERENCE",-1)]-aucpoint[(g,s)]})
    cells=pd.DataFrame(cell_rows).merge(elig,on="hospital_id",how="left").merge(zdw,on="hospital_id",how="left")
    cells.to_parquet(HC/"u2_hospital_seed_eul_cells.parquet",index=False)
    pd.DataFrame(cov_rows).to_parquet(HC/"u2_hospital_sampling_covariance_blocks.parquet",index=False)
    pd.DataFrame(manifests).to_csv(HC/"u2_hospital_sampling_covariance_manifest.csv",index=False)

    threshold_rows=[]; primary_rows=[]; blups=[]; varrows=[]; dwrows=[]; dwthr=[]; tabrows=[]
    fits_unadj={}
    for g in GENERATORS:
      for m in MODELS:
        allseeds=seedsets[g]
        for t,col in THRESHOLDS.items():
            hs=set(elig.loc[elig[col],"hospital_id"].astype(int)); d=cells[(cells.generator==g)&(cells.learner==m)&cells.hospital_id.isin(hs)].copy()
            covs={h:covdict[(g,m,h)] for h in hs}; fit=fit_reml(d,covs,False,allseeds); fits_unadj[(g,m,t)]=fit
            row=result_row(g,m,t,"PRIMARY_ALL_ESTIMABLE",fit); threshold_rows.append(row)
            if t=="200_20_20":
                primary_rows.append(row); varrows.append(row)
                for hid,v in fit["blup"].items():
                    raw=float(d[d.hospital_id==hid].EUL.mean()); eb=row["mu_EUL"]+v
                    blups.append({"hospital_id":hid,"generator":g,"learner":m,"raw_hospital_mean_EUL":raw,"hospital_random_effect_BLUP":v,"EB_hospital_EUL":eb,"shrinkage_magnitude":eb-raw,"raw_positive":raw>0,"EB_positive":eb>0})
            fitd=fit_reml(d,covs,True,allseeds); rr={"generator":g,"learner":m,"threshold":t,"seed_scope":"PRIMARY_ALL_ESTIMABLE","hospital_n":fitd["n_hospitals"],"seed_n":fitd["n_seeds"],
                "beta0":fitd["beta"][0],"beta_DW":fitd["beta"][1],"beta_DW_SE":fitd["se"][1],"beta_DW_CI_lower":fitd["beta"][1]-1.96*fitd["se"][1],"beta_DW_CI_upper":fitd["beta"][1]+1.96*fitd["se"][1],"beta_DW_p":2*norm.sf(abs(fitd["beta"][1]/fitd["se"][1])),
                "sigma2_seed_unadjusted":fit["sigma2_seed"],"sigma2_seed_adjusted":fitd["sigma2_seed"],"sigma2_hospital_unadjusted":fit["sigma2_hospital"],"sigma2_hospital_adjusted":fitd["sigma2_hospital"],"sigma2_interaction_unadjusted":fit["sigma2_interaction"],"sigma2_interaction_adjusted":fitd["sigma2_interaction"],
                "R2_hospital_DW":1-fitd["sigma2_hospital"]/fit["sigma2_hospital"] if fit["sigma2_hospital"]>0 else np.nan,"method":"CUSTOM_EXACT_GAUSSIAN_REML_KNOWN_BLOCK_V","converged":fitd["converged"]}
            dwthr.append(rr)
            if t=="200_20_20": dwrows.append(rr)
        if g=="TabDDPM":
            hs=set(elig.loc[elig.eligible_200_20_20,"hospital_id"].astype(int)); seeds=adequate[m]
            d=cells[(cells.generator==g)&(cells.learner==m)&cells.hospital_id.isin(hs)&cells.seed.isin(seeds)].copy()
            fit=fit_reml(d,{h:covdict[(g,m,h)] for h in hs},False,allseeds); tabrows.append(result_row(g,m,"200_20_20","TABDDPM_P_SI_GE_0_55",fit)|{"included_seeds":";".join(map(str,seeds))})
    pd.DataFrame(primary_rows).to_csv(HC/"u2_crossed_eul_primary_results.csv",index=False)
    pd.DataFrame(threshold_rows).to_csv(HC/"u2_crossed_eul_threshold_sensitivity.csv",index=False)
    pd.DataFrame(tabrows).to_csv(HC/"u2_crossed_eul_tabddpm_psige055.csv",index=False)
    pd.DataFrame(blups).to_csv(HC/"u2_hospital_blup_results.csv",index=False)
    pd.DataFrame(varrows).to_csv(HC/"u2_variance_component_summary.csv",index=False)
    pd.DataFrame(dwrows).to_csv(DD/"u2_crossed_DW_meta_regression.csv",index=False)
    pd.DataFrame(dwthr).to_csv(DD/"u2_crossed_DW_threshold_sensitivity.csv",index=False)
    counts=pd.DataFrame(blups).groupby(["generator","learner"]).agg(hospital_n=("hospital_id","size"),raw_positive_n=("raw_positive","sum"),EB_positive_n=("EB_positive","sum")).reset_index()
    counts.to_csv(HC/"u2_positive_hospital_descriptive_counts.csv",index=False)
    rng=pd.read_csv(QC/"U2_BOOTSTRAP_RNG_REGISTRY.csv"); extra=[]
    for mi,m in enumerate(MODELS):
        for hi,hid in enumerate(eligible_h): extra.append({"module":"hospital_paired_patient_covariance","generator":"ALL","domain":f"EICU_HOSPITAL_{hid}_{m}","rng_seed":BASE_SEED+mi*1000+hi,"B":B})
    pd.concat([rng,pd.DataFrame(extra)],ignore_index=True).to_csv(QC/"U2_BOOTSTRAP_RNG_REGISTRY.csv",index=False)
    qc={"B_cov":B,"covariance_blocks":len(manifests),"all_valid_B_ge_1000":bool(pd.DataFrame(manifests).valid_B.ge(1000).all()),"all_covariance_blocks_psd":bool(pd.DataFrame(manifests).min_eigen_after.ge(-1e-12).all()),"hospital_counts":{k:int(elig[v].sum()) for k,v in THRESHOLDS.items()},"hospital_ITL_crossed_status":"HOSPITAL_ITL_CROSSED_INFERENCE_NOT_ESTIMATED","hospital_ITL_reason":"Full joint MIMIC-plus-all-hospital covariance was computationally prohibitive; no simplification used.","custom_REML_reason":"metafor unavailable locally; exact Gaussian REML implemented with known block V and prespecified crossed random intercept covariance."}
    (QC/"U2_HOSPITAL_CROSSED_QC.json").write_text(json.dumps(qc,indent=2)+"\n")


if __name__=="__main__": main()
