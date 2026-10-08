"""Authenticated source excerpt for inspection only, not a standalone producer.
Global contexts belong to the original source; tested point adapter is hospital.py.
"""
from __future__ import annotations

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
