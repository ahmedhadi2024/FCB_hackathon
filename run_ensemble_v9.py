"""v9e: an average of several v9 networks. External data only.

Why: on real broadcast tracking (SkillCorner) individually trained v9-type networks score between
0.48 and 0.53; the spread comes from the training run, not from the method, and it decides
comparisons that look like method effects. Averaging the probabilities of several networks
before decoding removes most of that spread (three averaged: 0.538 on ten matches, above the
best single one).

Members are the shipped v9 plus networks trained identically (same nine matches, scenarios,
player faults and budget) with different seeds. The bundle stores them; inference averages
their team-swap-averaged probabilities and decodes once with v9's costs and filter.

    python run_ensemble_v9.py train 4      # four extra seeds, about 45 minutes
"""
import sys
from pathlib import Path
import torch
import run_retrain_v8 as v8
import run_retrain_v9 as v9
from fcb_external import retrain_v4, temporal
from fcb_external.feature_recovery_experiment import atomic_checkpoint
from fcb_external.schema import digest_file

ROOT=Path(__file__).resolve().parent
WEIGHT=3


def train(extra):
    retrain_v4.changed=v9.changed
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261117,4)
    out=ROOT/'models/temporal_v9e'; out.mkdir(parents=True,exist_ok=True)
    bundle=ROOT/'models/temporal_v9e.pt'
    if bundle.exists():
        raise FileExistsError('Ensemble bundle already exists')
    matches=v8.DFL+v8.GAMES
    seeds=[9106+i for i in range(extra)]
    todo=[s for s in seeds if not (out/f'seed_{s}.pt').exists()]
    if todo:
        arrays,_=retrain_v4.training_arrays(ROOT,cfg,matches,'final')
        arrays=v8.weighted(arrays,matches,WEIGHT)
        for seed in todo:
            ck=retrain_v4.fit(arrays,seed,cfg['network'],cfg['training'],module)
            ck.update(selected_transition_costs=list(v8.COSTS),min_controlled_frames=v8.FILTER,metrica_weight=WEIGHT)
            atomic_checkpoint(out/f'seed_{seed}.pt',ck); print('member saved',seed,flush=True)
        del arrays
    load=lambda path:torch.load(path,map_location='cpu',weights_only=True)
    members=[load(ROOT/'models/temporal_v9.pt')]+[load(out/f'seed_{s}.pt') for s in seeds]
    keep=('network','features','feature_module','mean','scale','state_dict','seed')
    atomic_checkpoint(bundle,{'format_version':1,'kind':'pytorch_temporal_ensemble_v1',
        'members':[{k:m[k] for k in keep} for m in members],'feature_module':members[0]['feature_module'],
        'features':members[0]['features'],'selected_transition_costs':list(v8.COSTS),'min_controlled_frames':v8.FILTER,
        'member_sha256':[digest_file(ROOT/'models/temporal_v9.pt')]+[digest_file(out/f'seed_{s}.pt') for s in seeds],
        'status':'v9e: average of v9 and identically trained networks with other seeds; no held-out score exists for this exact bundle'})
    print('Ensemble bundle written with',len(members),'members')


if __name__=='__main__':
    train(int(sys.argv[2]))
