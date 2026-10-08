"""Fabricated IO and AST-only regression tests; never execute a study driver."""
import ast
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import duckdb
ROOT=Path(__file__).resolve().parents[1]
def tree(p):return ast.parse((ROOT/p).read_text())
class ExportTests(unittest.TestCase):
 def setUp(self):
  fn=next(n for n in tree('src/harmonization/build_eicu_contract.py').body if isinstance(n,ast.FunctionDef) and n.name=='export_contract')
  ns={'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-export-function>','exec'),ns);self.export=ns['export_contract'];self.con=duckdb.connect()
  self.con.execute("CREATE TABLE eicu_contract_raw AS SELECT 1 AS fabricated_id, 'fabricated' AS label")
 def tearDown(self):self.con.close()
 def test_requested_path_parent_content_repeat(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'new parent'/"quoted'path.parquet";self.assertEqual(self.export(self.con,p),p)
   self.assertEqual(self.con.execute('SELECT * FROM read_parquet(?)',[str(p)]).fetchall(),[(1,'fabricated')])
   self.export(self.con,p);self.assertEqual(self.con.execute('SELECT * FROM read_parquet(?)',[str(p)]).fetchall(),[(1,'fabricated')])
   self.assertFalse((Path(d)/'{OUTPUT_PARQUET}').exists());self.assertFalse((ROOT/'{OUTPUT_PARQUET}').exists())
 def test_placeholder_rejected(self):
  for p in ['{OUTPUT_PARQUET}','','x\x00y']:
   with self.assertRaises(ValueError):self.export(self.con,p)
 def test_directory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.export(self.con,d)
class ScheduleTests(unittest.TestCase):
 def test_actual_loop_budget_and_lr(self):
  tr=tree('src/generators/run_tabddpm.py');epoch=next(n for n in ast.walk(tr) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch');inner=next(n for n in epoch.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
  assignments={}
  for n in ast.walk(tr):
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):assignments.setdefault(n.targets[0].id,n)
  ns=dict(n=40745,BATCH=256,N_EPOCHS=100,LR=1e-3,opt=SimpleNamespace(param_groups=[{'lr':1e-3}]),rates=[]);body=[]
  for n in inner.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'frac_done','executed_lr','executed_lr_min'}:
    body.append(copy.deepcopy(n))
    if n.targets[0].id=='executed_lr':body.append(ast.parse('rates.append(executed_lr)').body[0])
   elif isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='pg':body.append(copy.deepcopy(n))
   elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id in {'step','negative_lr_updates','zero_lr_updates'}:body.append(copy.deepcopy(n))
  loop=copy.deepcopy(inner);loop.body=body;outer=copy.deepcopy(epoch);outer.body=[loop]
  init=[copy.deepcopy(assignments[x]) for x in ['steps_per_epoch','total_steps','step','executed_lr_min','negative_lr_updates','zero_lr_updates']]
  exec(compile(ast.fix_missing_locations(ast.Module(body=init+[outer],type_ignores=[])),'<actual-scheduler-only>','exec'),ns)
  self.assertEqual(ns['steps_per_epoch'],math.ceil(40745/256));self.assertEqual(ns['steps_per_epoch'],160);self.assertEqual(ns['step'],16000);self.assertEqual(len(ns['rates']),16000);self.assertTrue(all(x>0 for x in ns['rates']));self.assertEqual(ns['negative_lr_updates'],0);self.assertEqual(ns['zero_lr_updates'],0);self.assertAlmostEqual(ns['rates'][-1],6.25e-8,delta=1e-18)
 def test_diffusion_sampling_configuration(self):
  tr=tree('src/generators/run_tabddpm.py');n=next(n for n in ast.walk(tr) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='GaussianMultinomialDiffusion');k={x.arg:ast.unparse(x.value) for x in n.keywords};self.assertEqual(k['scheduler'],"'cosine'")
  s=(ROOT/'src/generators/run_tabddpm.py').read_text();self.assertIn('FINAL_NON_EMA',s);self.assertIn('weight_decay=1e-4',s);self.assertIn('QuantileTransformer',s)
class LearnerTests(unittest.TestCase):
 def test_frozen_configuration(self):
  tr=tree('src/models/frozen_learners.py');c={n.func.id:n for n in ast.walk(tr) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['LogisticRegression','XGBClassifier']}
  self.assertEqual({x.arg:ast.literal_eval(x.value) for x in c['LogisticRegression'].keywords},dict(penalty='l2',C=1.0,max_iter=5000,solver='lbfgs',class_weight=None,random_state=42))
  self.assertEqual({x.arg:ast.literal_eval(x.value) for x in c['XGBClassifier'].keywords},dict(n_estimators=200,max_depth=5,learning_rate=0.05,min_child_weight=1,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',random_state=42,n_jobs=4))
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='StandardScaler' for n in ast.walk(tr)))
if __name__=='__main__':unittest.main()
