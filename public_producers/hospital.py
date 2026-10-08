"""Registry-to-hospital point-cell adapter for fabricated tests or authorized operators.
Frozen source point statements retained; no bootstrap or crossed fit is called.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def hospital_cells(registry, artifact_root):
    required={'generator','seed','model','domain','prediction_artifact','prediction_sha256'}
    if not required.issubset(registry.columns): raise ValueError('Prediction registry schema missing columns')
    reg=registry[registry.domain=='EICU_EXTERNAL']
    rows=[]
    for model,sub in reg.groupby('model',observed=True):
        refs=sub[sub.generator=='REAL_REFERENCE']
        if len(refs)!=1: raise ValueError('Exactly one real reference per learner required')
        import hashlib
        def read(row):
            p=Path(artifact_root)/row.prediction_artifact
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row.prediction_sha256: raise ValueError('Prediction file identity mismatch')
            d=pd.read_parquet(p).sort_values('evaluation_row_id').reset_index(drop=True)
            if not d.evaluation_row_id.is_unique or not d.patient_id_or_stay_key.is_unique: raise ValueError('Duplicate participant or row keys')
            if not d.predicted_probability.between(0,1).all() or not set(d.y_true).issubset({0,1}): raise ValueError('Invalid prediction or outcome domain')
            return d
        real=read(next(refs.itertuples(index=False)))
        syn=sub[sub.generator!='REAL_REFERENCE']
        if syn.duplicated(['generator','seed']).any(): raise ValueError('Duplicate realization registry key')
        for entry in syn.itertuples(index=False):
            sd=read(entry)
            for col in ['evaluation_row_id','patient_id_or_stay_key','hospital_id','y_true']:
                if not np.array_equal(real[col].to_numpy(),sd[col].to_numpy()): raise ValueError('Identity/order/pairing mismatch')
            for hid, rr in real.groupby('hospital_id',observed=True):
                ix=rr.index;yy=rr.y_true.to_numpy();pp=np.column_stack([rr.predicted_probability.to_numpy(),sd.loc[ix,'predicted_probability'].to_numpy()])
                labels=[('REAL_REFERENCE',-1),(entry.generator,int(entry.seed))]
                aucpoint={lab:roc_auc_score(yy,pp[:,j]) for j,lab in enumerate(labels)}
                g,s=entry.generator,int(entry.seed)
                rows.append({'hospital_id':hid,'generator':g,'learner':model,'seed':s,'N':len(ix),'events':int(yy.sum()),'non_events':int(len(ix)-yy.sum()),
                             'AUC_RE_h':aucpoint[('REAL_REFERENCE',-1)],'AUC_SE_sh':aucpoint[(g,s)],'EUL':aucpoint[('REAL_REFERENCE',-1)]-aucpoint[(g,s)],
                             'eligible_200_20_20':len(ix)>=200 and int(yy.sum())>=20 and int(len(ix)-yy.sum())>=20})
    return pd.DataFrame(rows)
