from pathlib import Path
import json,hashlib,csv
root=Path(__file__).resolve().parent/'study'
manifest=json.loads((root/'SHA256_MANIFEST.json').read_text())
for name,expected in manifest.items():
    p=root/Path(name.replace('\\','/'))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
rows=list(csv.DictReader((root/'neural network by python/final_unseen_1800/per_run.csv').open()))
assert len(rows)==150
expected={'ordinary':14,'predictions':23,'library':20,'hybrid':23}
for method,n in expected.items():
    assert sum(r['passed']=='True' for r in rows if r['method']==method)==n
assert all(int(r['combined_passes'])==0 for r in rows)
assert sum(int(r['evaluations']) for r in rows)==313920
print('PASS:',len(manifest),'frozen file hashes; 150 final runs; all reported pass counts.')
