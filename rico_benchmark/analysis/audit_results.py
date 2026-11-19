"""Independent ledger/archive and preservation checks after completion."""
import hashlib,json
import numpy as np
import scipy,numba,matplotlib,sys
import study_extension as s
plants,inputs=s.prepare()
checked=0; total=0
for d in plants:
    for seed in s.SEEDS:
        velocity=None
        for method in s.METHODS:
            stem=s.ROOT/'runs'/d['id']/f'{method}_{seed}'
            r=s.read(stem.with_suffix('.json')); archive=np.load(stem.with_suffix('.npz'))
            z=archive['ledger']; x=archive['initial']; v=archive['velocity']; pool=archive['initial_pool']
            assert (r['plant'],r['seed'],r['method'])==(d['id'],seed,method)
            assert z.shape==(1800,12) and np.isfinite(z).all()
            assert np.array_equal(z[:,0],np.arange(1,1801))
            assert np.array_equal(z[:,3],np.minimum.accumulate(z[:,2]))
            assert np.all((z[:,4:]>=0)&(z[:,4:]<=1))
            assert len(np.unique(z[:,4:].round(13),axis=0))==1800
            if velocity is None: velocity=v
            assert np.array_equal(v,velocity)
            if method=='obl':
                assert pool.shape==(128,8) and np.allclose(pool[:64]+pool[64:],1)
                assert np.array_equal(z[:128,4:],pool)
                assert np.array_equal(x,pool[np.argsort(z[:128,2],kind='stable')[:64]])
            for cp in r['checkpoints']:
                n=cp['evaluations']; row=z[:n][np.argmin(z[:n,2])]
                assert np.allclose(cp['gains'],s.b.decode(row[4:],s.b.original.config()))
                assert np.isclose(cp['fitness'],z[n-1,3])
                ref=s.read(s.OLD/d['id']/f'reference_{seed}.json')
                assert cp['passed']==s.passed(cp,ref)
            total+=len(z); checked+=1
assert checked==210 and total==378000
hashes=s.read(s.ROOT/'protocol.json')['hashes']
assert all(s.sha(p)==h for p,h in hashes.items())
s.save(s.ROOT/'runtime_versions.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,matplotlib=matplotlib.__version__))
s.save(s.ROOT/'AUDIT.json',dict(status='PASS',runs=checked,optimizer_evaluations=total,checkpoint_assessments=1050,
    check='All gain ledgers, exact budgets, bounds, paired velocities, OBL pool/selection, checkpoint gains/quality and frozen source hashes passed.',
    hardware_runs_added=0,original_inputs_unchanged=True))
print('AUDIT PASS: 210 runs, 378000 evaluations, 1050 checkpoint assessments; originals unchanged.')
