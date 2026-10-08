"""05S exact column projection plus fail-closed publication guard.
No fitting or value recalculation. Never exports the individual hospital axis.
"""
from pathlib import Path
import json,re,math
import pandas as pd

SCHEMA=json.loads(Path(__file__).with_name('aggregate_schema.json').read_text())
FORBIDDEN_KEYS={'hospital_id','hospitalid','hospital_key','source_hospitalid','source_hospital_id',
                'patient_id','subject_id','stay_id','hadm_id','patientunitstayid','patient_id_or_stay_key',
                'hospital_identity_sha256','hospital_key_namespace_sha256','prediction_artifact','predicted_probability'}

def _safe_value(value):
    if isinstance(value,dict):
        if any(str(k).lower() in FORBIDDEN_KEYS for k in value):raise ValueError('Restricted nested identifier field')
        for v in value.values():_safe_value(v)
    elif isinstance(value,(list,tuple)):
        for v in value:_safe_value(v)
    elif isinstance(value,str):
        if re.search(r'\bHOSP_|\b(?:hospital_id|hospital_key|subject_id|stay_id|patient_id)\b',value,re.I):raise ValueError('Restricted identifier marker')
        if value.lstrip().startswith(('[','{')):
            try:nested=json.loads(value)
            except json.JSONDecodeError:raise ValueError('Invalid structured field')
            _safe_value(nested)


def project_approved(frames):
    projected={}
    for name,frame in frames.items():
        if name=='EMPIRICAL_CELL_DECISIONS':continue  # Entire restricted object withheld by 05S.
        if name not in SCHEMA:raise ValueError('Unapproved public object: '+name)
        rule=SCHEMA[name];cols=rule['columns'];allowed=set(cols)|set(rule['omit'])
        if set(frame.columns)-allowed:raise ValueError('Unapproved field in '+name)
        if not set(cols).issubset(frame.columns):raise ValueError('Missing approved field in '+name)
        if frame.columns.duplicated().any():raise ValueError('Duplicate export columns')
        out=frame.loc[:,cols].copy()  # Exact scientific values/rows/missingness; no arithmetic.
        for col in out.columns:
            for value in out[col]:
                if col not in {'seed_rate_distribution_json','failure_reasons'}:
                    if not pd.api.types.is_scalar(value):raise ValueError('Non-scalar in approved scalar field')
                    if isinstance(value,str) and value.lstrip().startswith(('[','{')):raise ValueError('Structured object in approved scalar field')
                _safe_value(value)
        if 'failure_reasons' in out:
            if name!='U6_01_FIT_ACCOUNTING':raise ValueError('Unapproved failure accounting object')
            for value in out.failure_reasons:
                values=json.loads(value)
                if not isinstance(values,list):raise ValueError('Invalid failure accounting list')
                for record in values:
                    if not isinstance(record,dict) or set(record)!={'error_type','error_message','count'}:raise ValueError('Unapproved failure accounting schema')
                    if record['error_type']!='RuntimeError' or record['error_message']!='optimizer nonconvergence: ABNORMAL: ':raise ValueError('Unapproved failure accounting message')
                    if type(record['count']) is not int or record['count']<0:raise ValueError('Invalid failure accounting count')
        if 'seed_rate_distribution_json' in out:
            allowed_nested={'seed','qualified_count','pair_denominator','discordant_pair_count','pair_discordance_rate'}
            for value in out.seed_rate_distribution_json:
                values=json.loads(value)
                if not isinstance(values,list) or any(not isinstance(v,dict) or set(v)!=allowed_nested for v in values):raise ValueError('Unapproved per-realization JSON schema')
                seen=set()
                for record in values:
                    for key in ['seed','qualified_count','pair_denominator','discordant_pair_count']:
                        if type(record[key]) is not int:raise ValueError('Non-integer per-realization count')
                    seed=record['seed']
                    if seed not in range(42,57) or seed in seen:raise ValueError('Invalid or duplicate frozen realization index')
                    seen.add(seed)
                    if not 0<=record['qualified_count']<=85 or record['pair_denominator']!=85*84//2 or not 0<=record['discordant_pair_count']<=record['pair_denominator']:raise ValueError('Invalid aggregate count domain')
                    rate=record['pair_discordance_rate']
                    if type(rate) not in (float,int) or not math.isfinite(rate) or not 0<=rate<=1:raise ValueError('Non-scalar or invalid aggregate rate')
        projected[name]=out
    return projected


def export_approved(frames,destination):
    projected=project_approved(frames)  # Validate every frame before writing any output.
    root=Path(destination)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Nonempty output directory protected')
    root.mkdir(parents=True,exist_ok=True)
    for name,frame in projected.items():frame.to_csv(root/f'{name}.csv',index=False,float_format='%.17g')
    return projected
