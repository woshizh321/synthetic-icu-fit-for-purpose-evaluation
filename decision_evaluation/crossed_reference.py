"""Reference Gaussian crossed REML with supplied sampling covariance."""
import math
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

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
