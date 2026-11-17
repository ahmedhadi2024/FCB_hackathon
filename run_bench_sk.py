"""SkillCorner-only comparison of bundles (game 3 is training data for v10e, so Metrica 3 is not a test)."""
import sys,numpy as np,torch
import run_bench as rb, run_retrain_v8 as v8
from fcb_external import retrain_v4,temporal
cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261117,4)
models={n:torch.load(rb.ROOT/'models'/f'{n}.pt',map_location='cpu',weights_only=True) for n in sys.argv[1:]}
cells=rb.natural(models,module)
for n in models:
    r=[c for c in cells if c['model']==n]; print(n,'SkillCorner %.4f'%np.mean([c['spellF1'] for c in r]),'matches',len(r))
a,b=[{c['match']:c['spellF1'] for c in cells if c['model']==n} for n in list(models)[:2]]
print('second better on',sum(b[m]>a[m] for m in a),'of',len(a))
