"""Verify frozen publication sources; do not recompute scientific analyses."""
from pathlib import Path
import csv, hashlib
import pandas as pd
from public_producers.export import project_approved
ROOT=Path(__file__).resolve().parents[1]/'publication_data'

def test_release_data_manifest_exact_hashes():
    rows=list(csv.DictReader((ROOT/'MANIFEST.csv').open()))
    assert len(rows)==18
    for r in rows:
        p=ROOT.parent/r['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        assert p.stat().st_size==int(r['bytes'])

def test_all_fifteen_approved_source_projections():
    files=list((ROOT/'aggregate_sources').glob('*.csv'))
    assert len(files)==15
    frames={p.stem:pd.read_csv(p,keep_default_na=False) for p in files}
    projected=project_approved(frames)
    assert set(projected)==set(frames)
    for name in frames:
        pd.testing.assert_frame_equal(projected[name],frames[name],check_exact=True)

def test_frozen_unmet_overall_robustness_preserved():
    rows=list(csv.DictReader((ROOT/'aggregate_sources/U6_THRESHOLD_GATE.csv').open()))
    assert len(rows)==4
    assert sum(r['QUALIFYING_THRESHOLD']=='True' for r in rows)==2

def test_frozen_empirical_instability_source_coverage():
    rows=list(csv.DictReader((ROOT/'aggregate_sources/EMPIRICAL_MATRIX_SUMMARY.csv').open()))
    assert len(rows)==24
    assert all(r['both_states_present']=='True' for r in rows)
    assert all(int(r['qualified_cells'])+int(r['not_qualified_cells'])==int(r['total_cells']) for r in rows)
    assert all(float(r['hospital_reversal_proportion'])>0 and float(r['seed_reversal_proportion'])>0 for r in rows)


def test_failure_accounting_rejects_identifiers_vectors_and_new_fields():
    import json, pytest
    frame=pd.read_csv(ROOT/'aggregate_sources/U6_01_FIT_ACCOUNTING.csv',keep_default_na=False)
    for bad in [
        [{'error_type':'RuntimeError','error_message':'hospital_id 123','count':1}],
        [{'error_type':'RuntimeError','error_message':'optimizer nonconvergence: ABNORMAL: ','count':[1,2]}],
        [{'error_type':'RuntimeError','error_message':'optimizer nonconvergence: ABNORMAL: ','count':1,'hospital_id':123}],
    ]:
        changed=frame.copy()
        changed.loc[0,'failure_reasons']=json.dumps(bad)
        with pytest.raises(ValueError):project_approved({'U6_01_FIT_ACCOUNTING':changed})
