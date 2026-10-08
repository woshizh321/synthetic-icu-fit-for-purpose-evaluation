"""Fabricated-only numerical receipt. No clinical sources, clinical model fitting or changed expectations."""
import argparse,sys,json,hashlib,platform,os,io,contextlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--jac',choices=['original','3-point'],default='original');args=p.parse_args()
root=Path(args.repo).resolve();sys.path.insert(0,str(root))
import numpy as np, scipy,scipy.optimize
from threadpoolctl import threadpool_info
from portable_qualification import crossed_reml
from portable_qualification.cli import _fit_summary
from portable_qualification.algebra import qualify_summary
trace=[];original=crossed_reml.minimize

def instrumented(fun,x0,*a,**kw):
    if args.jac=='3-point':kw['jac']='3-point'
    result=original(fun,x0,*a,**kw)
    steps=[1e-4,1e-5,1e-6,1e-7,1e-8]
    gradients={str(h):[(fun(result.x+np.eye(3)[i]*h)-fun(result.x-np.eye(3)[i]*h))/(2*h) for i in range(3)] for h in steps}
    trace.append({'gradient_method':kw.get('jac'), 'optimizer_options':kw.get('options'), 'start':x0.tolist(),'theta':result.x.tolist(),'objective':float(result.fun),'jac_solver':result.jac.tolist(),'success':bool(result.success),'message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'central_gradients_by_step':gradients})
    return result
crossed_reml.minimize=instrumented
fixture=root/'portable_demo/example'
fit=_fit_summary(fixture/'fixture_cells.csv',fixture/'fixture_sampling_covariance.csv',fixture/'fixture_reliability.json')
card,_=qualify_summary(fit,m_max=20,delta=.15)
expected_file=fixture/'fixture_expected_qualification.json';expected=json.loads(expected_file.read_text())['expected_evidence_card']
diffs=[]
def walk(o,e,path=''):
    if isinstance(e,dict):
        for k in e:walk(o[k],e[k],path+'.'+k)
    elif isinstance(e,list):
        for i,(a,b) in enumerate(zip(o,e)):walk(a,b,path+f'[{i}]')
    elif isinstance(e,(int,float)) and not isinstance(e,bool):
        error=abs(float(o)-float(e));diffs.append({'path':path,'expected':e,'observed':o,'absolute_error':error,'relative_error':error/abs(e) if e else None})
walk(card,expected)
config=io.StringIO()
with contextlib.redirect_stdout(config):np.show_config();scipy.show_config()
receipt={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'dtype':str(np.dtype(float)),'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'threadpool':threadpool_info(),'numeric_library_config':config.getvalue(),'jac_mode':args.jac,'fit':fit,'card':card,'numeric_differences':sorted(diffs,key=lambda x:x['absolute_error'],reverse=True),'first_failed_assertion_in_frozen_order':next((d for d in diffs if d['absolute_error']>1e-6),None),'solver_trace':trace,'fixture_hashes':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in fixture.iterdir() if q.is_file()},'source_file_sha256':hashlib.sha256((root/'portable_qualification/crossed_reml.py').read_bytes()).hexdigest(),'clinical_data_used':False,'expected_value_modified':False,'tolerance':1e-6}
Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'versions':[receipt['python'],receipt['numpy'],receipt['scipy']],'max_difference':receipt['numeric_differences'][0],'first_failed':receipt['first_failed_assertion_in_frozen_order']},indent=2))
