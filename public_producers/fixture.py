"""Completely invented registry-to-aggregate fixture; no models trained."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .prediction import configure_prediction_output,write_prediction
from .hospital import hospital_cells
from .empirical import analyze_empirical
from .export import export_approved


def build_fixture(root, reference='mixed', complete=False):
    root=Path(root);configure_prediction_output(root/'predictions')
    ids=list(range(900001,900086));seeds=list(range(42,56));n=200
    hosp=np.repeat(ids,n);keys=np.arange(len(hosp))+8000000;y=np.tile(np.r_[np.zeros(n//2),np.ones(n//2)],len(ids)).astype(np.int8)
    records=[]
    generators={'TabDDPM':seeds}
    if complete:generators={'GaussianCopula':list(range(42,57)),'CTGAN':list(range(42,57)),'TabDDPM':seeds}
    cases=[('REAL_REFERENCE',None)]+[(g,s) for g,ss in generators.items() for s in ss]
    for model in ['LR_L2','XGBoost']:
        for generator,seed in cases:
            probability=y.astype(float).copy()
            if seed is not None:
                for i in range(len(ids)):
                    flips=((seed-42+i)%6)*20
                    probability[i*n+n//2:i*n+n//2+flips]=0.0
            p=write_prediction(generator,seed,model,'EICU_EXTERNAL',keys,hosp,y,probability)
            records.append({'generator':generator,'seed':seed,'model':model,'domain':'EICU_EXTERNAL','n':len(y),'events':int(y.sum()),'auroc':float(roc_auc_score(y,probability)),
                            'prediction_artifact':str(p.relative_to(root)),'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':'FABRICATED_NO_CLINICAL_DATA'})
    registry=pd.DataFrame(records);registry['seed']=registry.seed.astype('Int64')
    cells=hospital_cells(registry,root)
    refs=[];templates=[]
    bounds={'qualified':(-.1,.025),'not_qualified':(.21,.3),'indeterminate':(.03,.3),'mixed':(.03,.1)}
    lower,upper=bounds[reference]
    for generator,ss in generators.items():
        for model in ['LR_L2','XGBoost']:
            templates.append({'generator':generator,'learner':model,'seed_n':len(ss),'seed_membership':ss})
            refs.append({'generator':generator,'learner':model,'hospital_n':85,'seed_n':len(ss),'cell_n':len(ss)*85,'converged':True,
                         'new_seed_new_hospital_PI95_lower':lower,'new_seed_new_hospital_PI95_upper':upper})
    return registry,cells,pd.DataFrame(refs),templates,ids


def run_fixture(destination):
    destination=Path(destination)
    registry,cells,refs,templates,ids=build_fixture(destination/'restricted_fabricated_work',complete=True)
    outputs=analyze_empirical(cells,refs,templates,ids,[.025,.05,.1,.2],collapsed_seed_ids=[56])
    public=export_approved(outputs,destination/'public_aggregates')
    summary={'mode':'PUBLIC_AGGREGATE_AND_FABRICATED_REPRODUCTION','fixture_is_fabricated':True,'trained_models':0,'clinical_records_used':0,
             'registry_rows':len(registry),'hospital_cells':len(cells),'matrix_rows':len(outputs['EMPIRICAL_MATRIX_SUMMARY']),'public_objects':sorted(public),'historical_clinical_reconstruction_proved':False}
    (destination/'FIXTURE_QC.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Fabricated producer integration only; no restricted inputs')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run_fixture(args.output),indent=2))
