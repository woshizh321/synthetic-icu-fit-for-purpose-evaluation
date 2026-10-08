import unittest,tempfile,json,hashlib,ast
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from public_producers.fixture import build_fixture,run_fixture
from public_producers.empirical import analyze_empirical
from public_producers.hospital import hospital_cells
from public_producers.export import export_approved,project_approved,SCHEMA

class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.registry,cls.cells,cls.refs,cls.templates,cls.ids=build_fixture(cls.root)
        cls.outputs=analyze_empirical(cls.cells,cls.refs,cls.templates,cls.ids,[.025,.05,.1,.2],[56])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def analyze(self,cells=None,refs=None):
        return analyze_empirical(self.cells if cells is None else cells,self.refs if refs is None else refs,self.templates,self.ids,[.025,.05,.1,.2],[56])
    def test_01_registry_columns_and_dtypes(self):
        self.assertEqual(list(self.registry),['generator','seed','model','domain','n','events','auroc','prediction_artifact','prediction_sha256','source'])
        self.assertEqual(str(self.registry.seed.dtype),'Int64');self.assertEqual(str(self.registry.auroc.dtype),'float64')
        schema=pq.read_table(self.root/self.registry.iloc[1].prediction_artifact).schema
        self.assertEqual(schema.names,['generator','seed','model','domain','evaluation_row_id','patient_id_or_stay_key','hospital_id','y_true','predicted_probability','prediction_status'])
        self.assertEqual(str(schema.field('seed').type),'int64');self.assertEqual(str(schema.field('y_true').type),'int8');self.assertEqual(str(schema.field('predicted_probability').type),'double')
        self.assertTrue(str(schema.field('generator').type).startswith('dictionary'))
    def test_02_stage_schema_compatibility(self):
        self.assertEqual(len(self.cells),2*14*85)
        for name,rule in SCHEMA.items():
            if name in self.outputs:self.assertEqual(set(self.outputs[name].columns),set(rule['columns'])|set(rule['omit']))
    def test_03_EUL_is_real_minus_synthetic(self):
        np.testing.assert_array_equal(self.cells.EUL,self.cells.AUC_RE_h-self.cells.AUC_SE_sh)
        self.assertAlmostEqual(self.cells.iloc[0].AUC_RE_h,1.0)
        self.assertAlmostEqual(self.cells.iloc[1].EUL,.1)
    def test_04_equality_is_qualified(self):
        c=self.cells.copy();c['AUC_RE_h']=.5+.025;c['AUC_SE_sh']=.5;c['EUL']=.025
        out=self.analyze(c)['EMPIRICAL_CELL_DECISIONS'];self.assertTrue((out.binary_decision=='QUALIFIED').all())
    def test_05_strictly_above_is_not_qualified(self):
        c=self.cells.copy();c['AUC_RE_h']=.5+.20001;c['AUC_SE_sh']=.5;c['EUL']=.20001
        self.assertTrue((self.analyze(c)['EMPIRICAL_CELL_DECISIONS'].binary_decision=='NOT_QUALIFIED').all())
    def test_06_unordered_realization_pairs(self):
        out=self.outputs['REALIZATION_DISCORDANCE'];per=self.outputs['EMPIRICAL_CELL_DECISIONS']
        for r in out.itertuples():
            q=per[(per.learner==r.learner)&(per.delta==r.delta)]
            pairs=discord=0
            for _,sub in q.groupby('hospital_id'):
                vals=(sub.binary_decision=='QUALIFIED').tolist()
                for a,b in combinations(vals,2):pairs+=1;discord+=a!=b
            self.assertEqual(pairs,r.total_seed_pairs);self.assertEqual(discord,r.discordant_seed_pairs)
    def test_07_unordered_destination_pairs(self):
        out=self.outputs['DESTINATION_DISCORDANCE'];per=self.outputs['EMPIRICAL_CELL_DECISIONS']
        for r in out.itertuples():
            q=per[(per.learner==r.learner)&(per.delta==r.delta)]
            pairs=discord=0
            for _,sub in q.groupby('seed'):
                vals=(sub.binary_decision=='QUALIFIED').tolist()
                for a,b in combinations(vals,2):pairs+=1;discord+=a!=b
            self.assertEqual(pairs,r.total_hospital_pairs);self.assertEqual(discord,r.discordant_hospital_pairs)
    def test_08_reversal_partition_and_denominators(self):
        for r in self.outputs['REALIZATION_DISCORDANCE'].itertuples():
            self.assertEqual(r.hospital_reversal_count+r.unanimous_qualified_hospital_count+r.unanimous_not_qualified_hospital_count,85)
            self.assertEqual(r.total_seed_pairs,85*14*13//2)
        for r in self.outputs['DESTINATION_DISCORDANCE'].itertuples():
            self.assertEqual(r.seed_reversal_count+r.unanimous_qualified_seed_count+r.unanimous_not_qualified_seed_count,14)
            self.assertEqual(r.total_hospital_pairs,14*85*84//2)
        d=self.outputs['EMPIRICAL_DENOMINATORS'];self.assertTrue((d.expected_cells==14*85).all());self.assertTrue((d.collapsed_attempt_count==1).all())
    def test_09_TargetB_all_states_and_equality(self):
        r=self.refs.copy();r['new_seed_new_hospital_PI95_lower']=-.1;r['new_seed_new_hospital_PI95_upper']=.025
        out=self.analyze(refs=r)['TARGET_B_REFERENCE'];self.assertTrue((out.full_Target_B_state=='FULL_TARGET_B_QUALIFIED').all())
        r['new_seed_new_hospital_PI95_lower']=.21;r['new_seed_new_hospital_PI95_upper']=.3
        self.assertTrue((self.analyze(refs=r)['TARGET_B_REFERENCE'].full_Target_B_state=='FULL_TARGET_B_NOT_QUALIFIED').all())
        r['new_seed_new_hospital_PI95_lower']=.025;r['new_seed_new_hospital_PI95_upper']=.3
        out=self.analyze(refs=r)['TARGET_B_REFERENCE'];self.assertTrue((out.full_Target_B_state=='FULL_TARGET_B_INDETERMINATE').all())
    def test_10_indeterminate_NA_survives_CSV(self):
        r=self.refs.copy();r['new_seed_new_hospital_PI95_lower']=0.;r['new_seed_new_hospital_PI95_upper']=.3
        outputs=self.analyze(refs=r);out=outputs['TARGET_B_REFERENCE'];self.assertTrue(out.opposite_cell_count.isna().all());self.assertTrue(out.CELL_TO_TARGET_B_DISCORDANCE.isna().all())
        with tempfile.TemporaryDirectory() as p:
            export_approved(outputs,p);x=pd.read_csv(Path(p)/'TARGET_B_REFERENCE.csv');self.assertTrue(x.opposite_cell_count.isna().all());self.assertTrue(x.CELL_TO_TARGET_B_DISCORDANCE.isna().all())
    def test_11_executed_oracle_exact_equivalence(self):
        golden=Path(__file__).resolve().parents[1]/'public_producers/golden/executed_matrix_algorithm.py'
        with tempfile.TemporaryDirectory() as p:
            ns=dict(np=np,pd=pd,json=json,check=lambda ok,msg:self.assertTrue(bool(ok),msg),OUT=Path(p),
                    lock={'templates':self.templates,'delta_grid':[.025,.05,.1,.2],'hospital_identity_sha256':'CONTROLLED_HISTORICAL_IDENTITY_NOT_DISTRIBUTED'},
                    ids=self.ids,c=self.cells,cells=self.cells,tb=self.refs,collapsed_seed_ids=[56])
            exec(compile(golden.read_text(),str(golden),'exec'),ns)
            for name,key in [('U6_02_REALIZATION_INSTABILITY.csv','REALIZATION_DISCORDANCE'),('U6_02_DESTINATION_INSTABILITY.csv','DESTINATION_DISCORDANCE'),('U6_02_JOINT_MATRIX_SUMMARY.csv','EMPIRICAL_MATRIX_SUMMARY'),('U6_02_TARGET_B_REFERENCE_CONCORDANCE.csv','TARGET_B_REFERENCE'),('U6_02_DENOMINATOR_REPORT.csv','EMPIRICAL_DENOMINATORS')]:
                source_frame={'REALIZATION_DISCORDANCE':ns['realization'],'DESTINATION_DISCORDANCE':ns['destination'],'EMPIRICAL_MATRIX_SUMMARY':ns['mat'],'TARGET_B_REFERENCE':ns['reference'],'EMPIRICAL_DENOMINATORS':pd.DataFrame(ns['denoms'])}[key]
                pd.testing.assert_frame_equal(source_frame,self.outputs[key],check_exact=True)
                copy=Path(p)/('candidate_'+name);self.outputs[key].to_csv(copy,index=False,float_format='%.17g')
                self.assertEqual((Path(p)/name).read_bytes(),copy.read_bytes())
            pd.testing.assert_frame_equal(pd.read_parquet(Path(p)/'U6_02_PER_CELL_DECISIONS.parquet'),self.outputs['EMPIRICAL_CELL_DECISIONS'],check_exact=True)
    def test_12_export_withholds_individual_hospital_object(self):
        p=project_approved(self.outputs);self.assertNotIn('EMPIRICAL_CELL_DECISIONS',p)
        self.assertNotIn('hospital_rate_distribution_json',p['REALIZATION_DISCORDANCE']);self.assertNotIn('hospital_identity_sha256',p['EMPIRICAL_DENOMINATORS'])
        for name,frame in p.items():pd.testing.assert_frame_equal(frame,self.outputs[name][SCHEMA[name]['columns']],check_exact=True)
    def test_13_forbidden_extra_field_rejected(self):
        x=self.outputs['TARGET_B_REFERENCE'].copy();x['hospital_id']=123
        with self.assertRaises(ValueError):project_approved({'TARGET_B_REFERENCE':x})
    def test_14_nested_hospital_identifier_rejected(self):
        x=self.outputs['DESTINATION_DISCORDANCE'].copy();x['seed_rate_distribution_json']='[{"hospital_id":123}]'
        with self.assertRaises(ValueError):project_approved({'DESTINATION_DISCORDANCE':x})
    def test_15_unknown_object_rejected(self):
        with self.assertRaises(ValueError):project_approved({'PRIVATE_MAPPING':pd.DataFrame({'hospital_id':[123]})})
    def test_16_atomic_guard_no_output_on_rejection(self):
        x=self.outputs['TARGET_B_REFERENCE'].copy();x['patient_id']=7
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):export_approved({'EMPIRICAL_MATRIX_SUMMARY':self.outputs['EMPIRICAL_MATRIX_SUMMARY'],'TARGET_B_REFERENCE':x},Path(p)/'out')
            self.assertFalse((Path(p)/'out').exists())
    def test_17_missing_cartesian_cell_rejected(self):
        with self.assertRaises(ValueError):self.analyze(self.cells.iloc[1:])
    def test_18_duplicate_cell_rejected(self):
        with self.assertRaises(ValueError):self.analyze(pd.concat([self.cells,self.cells.iloc[:1]]))
    def test_19_nonfinite_EUL_rejected(self):
        x=self.cells.copy();x.loc[0,'EUL']=float('nan')
        with self.assertRaises(ValueError):self.analyze(x)
    def test_20_changed_tolerance_rejected(self):
        with self.assertRaises(ValueError):analyze_empirical(self.cells,self.refs,self.templates,self.ids,[.03,.05,.1,.2])
    def test_21_bad_reference_rejected(self):
        x=self.refs.copy();x.loc[0,'new_seed_new_hospital_PI95_lower']=.5
        with self.assertRaises(ValueError):self.analyze(refs=x)
    def test_22_pairing_mismatch_rejected(self):
        x=self.registry.copy();path=self.root/x.iloc[1].prediction_artifact
        original=path.read_bytes()
        try:
            d=pd.read_parquet(path);d.loc[0,'patient_id_or_stay_key']=-1;d.to_parquet(path,index=False);x.loc[1,'prediction_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):hospital_cells(x,self.root)
        finally:path.write_bytes(original)
    def test_23_prediction_file_hash_rejected(self):
        x=self.registry.copy();x.loc[1,'prediction_sha256']='0'*64
        with self.assertRaises(ValueError):hospital_cells(x,self.root)
    def test_24_complete_integration_no_restricted_output(self):
        with tempfile.TemporaryDirectory() as p:
            q=run_fixture(Path(p)/'fixture');self.assertEqual(q['trained_models'],0);self.assertEqual(q['clinical_records_used'],0);self.assertFalse(q['historical_clinical_reconstruction_proved'])
            self.assertEqual(len(list((Path(p)/'fixture/public_aggregates').glob('*.csv'))),5)
            self.assertEqual(q['registry_rows'],90);self.assertEqual(q['matrix_rows'],24)
    def test_26_nested_object_in_scalar_field_rejected(self):
        x=self.outputs['TARGET_B_REFERENCE'].copy();x['reference_lower_PI']='{"hospital_ids":[900001,900002],"EUL":[0.01,0.02]}'
        with self.assertRaises(ValueError):project_approved({'TARGET_B_REFERENCE':x})
    def test_27_vector_in_seed_JSON_scalar_rejected(self):
        x=self.outputs['DESTINATION_DISCORDANCE'].copy()
        x['seed_rate_distribution_json']='[{"seed":42,"qualified_count":[1,2],"pair_denominator":3570,"discordant_pair_count":1,"pair_discordance_rate":0.1}]'
        with self.assertRaises(ValueError):project_approved({'DESTINATION_DISCORDANCE':x})
    def test_25_input_cell_permutation_invariant(self):
        outputs=self.analyze(self.cells.iloc[::-1])
        for k in self.outputs:
            if k=='EMPIRICAL_CELL_DECISIONS':
                cols=['generator','learner','seed','hospital_id','delta']
                pd.testing.assert_frame_equal(outputs[k].sort_values(cols).reset_index(drop=True),self.outputs[k].sort_values(cols).reset_index(drop=True),check_exact=True)
            else:pd.testing.assert_frame_equal(outputs[k],self.outputs[k],check_exact=True)

if __name__=='__main__':unittest.main()
