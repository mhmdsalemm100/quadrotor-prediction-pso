import json,urllib.request,hashlib
from pathlib import Path
R=Path(r'SOURCE_HIL/firmware')
req=urllib.request.Request('https://api.github.com/repos/PX4/PX4-Autopilot/releases/tags/v1.17.0',headers={'User-Agent':'HIL-study-setup'})
with urllib.request.urlopen(req,timeout=30) as f:d=json.load(f)
a=next(x for x in d['assets'] if x['name']=='px4_fmu-v6c_default.px4')
with urllib.request.urlopen(a['browser_download_url'],timeout=60) as f:b=f.read()
meta=json.loads(b);assert meta['board_id']==56,meta['board_id']
p=R/'original_px4_fmu-v6c_default.px4';p.write_bytes(b)
report={'url':a['browser_download_url'],'board_id':meta['board_id'],'sha256':hashlib.sha256(b).hexdigest(),'purpose':'Official stock recovery image; not flashed'}
(R/'original_firmware_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
