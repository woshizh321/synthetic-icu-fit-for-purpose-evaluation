"""Verify release identities and public display values without fitting any model."""
from pathlib import Path
import argparse,ast,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(rendered=None):
    manifest=ROOT/'PUBLIC_RELEASE_MANIFEST_v1.4.0.csv'
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
    public_input=ROOT/'extension/e3_e5/public_inputs/V2_P95_REPLICATE_METRICS.parquet'
    assert sha(public_input)=='50b69c5b0d308b367f388508587cf6e6a927087342640e15dd74e2e8f214febd'
    supp=ROOT/'supplementary_data'
    expected_supp={
      1:'849d96486177d342b824edbd1e37e1d52bbc7fc5e83ad42f329695aa27fe3f57',
      2:'d7ea650d21d06bcaa5205313a492aa6c884cb0664422355b5c577601fb24ec93',
      3:'8b2780b45c4096df83c2a9f72ff2b3088f97ea5415f006df410084a3f329b953',
      4:'a4b195ea3b4567f16b2f614cd090cb3f417ddd660388d55991cb3627e1cfb115',
      5:'a14dab68110986cce26773075f2780022aa6225f746954c275ea02dd1579d4a7',
      6:'82733f18f1a6c6ea409435fdd4ec63dfd4e0fb483ff9aa0ab9c21e35fa642fdb',
      7:'d6cde787e0ad5e0628a530f8a1ae5f8a41cc74d5f389face24f95e0997e2af60',
      8:'ae502170372f6b166d4e5d23e5d70e7987f743a5d806292df3ead56e03626c00',
      9:'902c4bf85aadcee9b7f97d9d237ee956008275c94a25cb33560a4db4879c9bda',
      10:'09f8513defb65ee05fa09b3053cd045a18e39e15079d24e94359be363af0e063'}
    files=sorted(supp.glob('Supplementary_Data_*.csv'))
    assert len(files)==10
    for p in files:
      number=int(p.name.split('_')[2]);assert sha(p)==expected_supp[number],p.name
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
    print(json.dumps({'manifest_files_verified':len(rows),'aggregate_source_checks':'PASS','supplementary_data_1_10':'PASS','public_e3_e5_input':'PASS','rendered_displays':'PASS' if rendered else 'NOT_REQUESTED','scientific_analysis_rerun':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rendered',type=Path);args=p.parse_args();verify(args.rendered)
