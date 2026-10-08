"""Exact recovered historical prediction functions. No fits occur on import.
PREDS is an interface destination and must be configured by caller.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
PREDS = None

def configure_prediction_output(path):
    global PREDS
    PREDS = Path(path)

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def fit_lr(X,y):
    scaler=StandardScaler().fit(X)
    model=LogisticRegression(penalty="l2",C=1.0,max_iter=5000,solver="lbfgs",class_weight=None,random_state=42)
    model.fit(scaler.transform(X),y)
    return ("lr",model,scaler)

def fit_xgb(X,y):
    model=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,min_child_weight=1,
                        subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42,n_jobs=4)
    model.fit(X,y)
    return ("xgb",model,None)

def predict(mt,X):
    kind,model,scaler=mt
    return model.predict_proba(scaler.transform(X) if kind=="lr" else X)[:,1]

def safe_gen(g): return {"GaussianCopula":"gaussiancopula","CTGAN":"ctgan","TabDDPM":"tabddpm"}[g]

def legacy_rid(g,seed): return f"{safe_gen(g)}_repA_seed{seed}_N40745"

def write_prediction(generator,seed,model,domain,keys,hosp,y,p,status="ESTIMABLE"):
    out=PREDS/("real" if generator=="REAL_REFERENCE" else "synthetic")
    if generator!="REAL_REFERENCE": out=out/f"generator={generator}"/f"seed={seed}"
    out=out/f"model={model}"/f"domain={domain}"
    out.mkdir(parents=True,exist_ok=False)
    path=out/"predictions.parquet"
    n=len(y)
    df=pd.DataFrame({
        "generator":pd.Series([generator]*n,dtype="category"),
        "seed":pd.Series([seed]*n,dtype="Int64"),
        "model":pd.Series([model]*n,dtype="category"),
        "domain":pd.Series([domain]*n,dtype="category"),
        "evaluation_row_id":np.arange(n,dtype=np.int64),
        "patient_id_or_stay_key":keys,
        "hospital_id":hosp,
        "y_true":np.asarray(y,dtype=np.int8),
        "predicted_probability":np.asarray(p,dtype=np.float64),
        "prediction_status":pd.Series([status]*n,dtype="category"),
    })
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),path,compression="zstd",use_dictionary=True)
    return path
