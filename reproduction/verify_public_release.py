"""Verify release identities and public display values without fitting any model."""
from pathlib import Path
import argparse,ast,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(rendered=None):
    manifest=ROOT/'PUBLIC_RELEASE_MANIFEST_v1.3.0.csv'
    with manifest.open() as f: rows=list(csv.DictReader(f))
    assert len({r['path'] for r in rows})==len(rows)
    for r in rows:
        p=ROOT/r['path'];assert p.is_file() and sha(p)==r['sha256'],r['path']
        assert p.stat().st_size==int(r['size_bytes']) and r['public_release_status']=='APPROVED'
        assert r['contains_patient_level_data']=='false' and r['contains_restricted_derivative']=='false'
    for p in list((ROOT/'decision_evaluation').glob('*.py'))+list((ROOT/'reproduction').glob('*.py')):ast.parse(p.read_text())
    source=ROOT/'manuscript_source_data'
    def read(n):
        with (source/n).open() as f:return list(csv.DictReader(f))
    expected={'FIGURE2_SOURCE_DATA.csv':48,'FIGURE3_SOURCE_DATA.csv':192,'FIGURE4_SOURCE_DATA.csv':576,'FIGURE5A_SOURCE_DATA.csv':48,'FIGURE5B_SOURCE_DATA.csv':288,'FIGURES1_SOURCE_DATA.csv':240}
    for name,n in expected.items():assert len(read(name))==n
    coverage=read('FIGURE4_SOURCE_DATA.csv');assert min(float(r['coverage']) for r in coverage)==.5 and max(float(r['coverage']) for r in coverage)==1
    generation=read('approved_supplementary_aggregate_tables/TableS16_Generation.csv');assert {r['generator']:int(r['utility_estimable_n']) for r in generation}=={'GaussianCopula':15,'CTGAN':15,'TabDDPM':14};assert all(int(r['generation_attempted_n'])==15 for r in generation)
    assert sorted({float(r['confidence']) for r in read('FIGURES1_SOURCE_DATA.csv')})==[.8,.9,.95,.975,.99]
    if rendered:
        from PIL import Image,ImageChops
        for i in range(1,6):
            with Image.open(ROOT/f'manuscript_figures/Figure{i}.png') as a,Image.open(rendered/f'MAIN_FIGURES/Figure{i}.png') as b:
                assert a.size==b.size,(i,a.size,b.size)
                assert ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox() is None,f'Figure{i} pixels differ'
        data=json.loads((rendered/'QC/FINAL_DISPLAY_RENDER_DATA.json').read_text())
        assert len(data['coverage'])==576 and len(data['endpoint_changes'])==192 and len(data['classifications'])==288 and len(data['trajectories'])==960
        lookup={(r['dgp'],r['template_id'],r['regime'],r['strategy'],r['target']):float(r['coverage']) for r in coverage}
        assert all(abs(lookup[(r['dgp'],r['template_id'],r['regime'],r['strategy'],r['target'])]-r['coverage'])<1e-15 for r in data['coverage'])
    print(json.dumps({'manifest_files_verified':len(rows),'aggregate_source_checks':'PASS','rendered_displays':'PASS' if rendered else 'NOT_REQUESTED','scientific_analysis_rerun':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rendered',type=Path);args=p.parse_args();verify(args.rendered)
