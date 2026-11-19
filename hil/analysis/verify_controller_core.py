from pathlib import Path
import sys,subprocess,json,io,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
S=Path(r'SOURCE_STUDY')
sys.path.insert(0,str(S));import model
rows=[]
for axis,amplitude in [(0,5),(0,20),(1,5),(1,20),(2,30),(2,100)]:
 target=np.zeros(3);target[axis]=np.deg2rad(amplitude);g=model.baseline()
 metrics,diag,trace=model.simulate(g,target,model.NOMINAL,.609,.001,.001,.004,.2,3.,store=True)
 frames=np.column_stack([trace[:,1:3],trace[:,4:7],trace[:,7:10],np.full(len(trace),.609)])
 text='\n'.join(' '.join(format(x,'.17g') for x in row) for row in frames)+'\n'
 cmd=['wsl.exe','-d','RflySim-20.04','--exec','/mnt/cUSER_HOME/Desktop/HIL of PSOand NN/firmware/controller_reference_cli']+[format(x,'.17g') for x in g]
 result=subprocess.run(cmd,input=text,text=True,capture_output=True,timeout=45,check=True)
 got=np.loadtxt(io.StringIO(result.stdout));expected=np.column_stack([trace[:,23:27],trace[:,13:16]])
 error=np.max(np.abs(got-expected));assert error<1e-10,(axis,amplitude,error)
 rows.append({'axis':axis,'amplitude_deg_or_deg_s':amplitude,'samples':len(trace),'max_absolute_error':float(error)})
report={'scope':'C++ host replay against frozen Python controller traces; not ARM hardware validation','passed':True,'cases':rows,'source_model_sha256':hashlib.sha256((S/'model.py').read_bytes()).hexdigest(),'controller_header_sha256':hashlib.sha256((R/'firmware/study_controller.hpp').read_bytes()).hexdigest()}
(R/'reports/controller_equivalence_host.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
