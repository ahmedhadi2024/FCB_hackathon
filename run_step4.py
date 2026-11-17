"""Step 4: Metrica sample game 3 as training data, and a larger seed average. External data only.

Protocol (fixed before any result):
  feats   cache v9-style training features (target-like ball loss + player faults, 7 scenarios) for
          the 7 DFL matches and Metrica games 1-3 on disk (memmaps; RAM is tight).
  logo    leave-one-Metrica-game-out, G in {1,2}. Condition A: DFL + the other of games 1/2 (w3).
          Condition B: A + game 3 (w3). 3 seeds each, GPU. Score the 3-seed average on G
          (6 scenarios x ball held/blank x players faults/all) and on 20 SkillCorner matches.
  rule    adopt game 3 only if mean over G of (B - A) on held+faults >= +0.005, no fold below -0.005,
          and SkillCorner (B ensemble vs A ensemble) not lower than -0.005.
  final   DFL + games 1-3 (w3), N seeds, bundle models/temporal_v10e.pt (kind ensemble_v1).

    FCB_DEVICE=cuda .venv-gpu python run_step4.py feats | logo | final N
"""
import sys,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import run_retrain_v8 as v8
import run_retrain_v9 as v9
import run_bench as rb
from fcb_external import metrica_sample,retrain_v4,temporal,tracking_faults
from fcb_external.feature_recovery_experiment import atomic_checkpoint,seed_for
from fcb_external.schema import Targets,digest_file,write_json

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'artifacts/cache_step4'
GAMES3=['Sample_Game_1','Sample_Game_2','Sample_Game_3']
ALL=v8.DFL+GAMES3
WEIGHT=3


def prepare_game3():
    for name,(obs,label) in metrica_sample.load_all(ROOT/'external-data/metrica_sample',include_game_3=True).items():
        folder=ROOT/'artifacts'/name
        if (folder/'observations/metadata.json').exists():
            continue
        held=v8.hold_ball(obs); held.save(folder/'observations'); Targets(label,label!='X').save(folder/'targets',held)
        print('Prepared',name,flush=True)


def feats():
    retrain_v4.changed=v9.changed
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261117,4)
    prepare_game3(); CACHE.mkdir(parents=True,exist_ok=True)
    scenarios=[s for s in cfg['scenarios'] if s['name'] in cfg['training_scenarios']]
    for match in ALL:
        obs,targets=retrain_v4.load_external(ROOT,match)
        y=retrain_v4.integer_labels(retrain_v4.convention_labels(cfg,targets,obs)); seq=np.array(obs.sequence_id)
        np.save(CACHE/f'{match}_y.npy',y); np.save(CACHE/f'{match}_seq.npy',seq)
        for scenario in scenarios:
            path=CACHE/f'{match}_{scenario["name"]}_x.npy'
            if path.exists():
                continue
            x=module.extract(retrain_v4.changed(obs,scenario,'train','final'))
            np.save(str(path)+'.tmp.npy',x); (CACHE/(path.name+'.tmp.npy')).replace(path); del x
            print('features',match,scenario['name'],flush=True)
    print('feature cache ready')


def arrays_for(matches,weight=WEIGHT):
    cfg=v8.config(); out=[]
    for match in matches:
        y=np.load(CACHE/f'{match}_y.npy'); seq=np.load(CACHE/f'{match}_seq.npy')
        group=[(np.load(CACHE/f'{match}_{name}_x.npy',mmap_mode='r'),y,seq) for name in cfg['training_scenarios']]
        out+=group*(weight if match in GAMES3 else 1)
    return out


def train_models(matches,seeds,tag):
    cfg=v8.config(); module=retrain_v4.module_for(cfg); out=[]
    for seed in seeds:
        path=ROOT/f'models/temporal_step4/{tag}_{seed}.pt'
        if path.exists():
            out.append(torch.load(path,map_location='cpu',weights_only=True)); continue
        path.parent.mkdir(parents=True,exist_ok=True)
        ck=retrain_v4.fit(arrays_for(matches),seed,cfg['network'],cfg['training'],module)
        ck.update(selected_transition_costs=list(v8.COSTS),min_controlled_frames=v8.FILTER,metrica_weight=WEIGHT)
        atomic_checkpoint(path,ck); out.append(torch.load(path,map_location='cpu',weights_only=True))
        print('trained',tag,seed,flush=True)
    return out


def bundle(members):
    keep=('network','features','feature_module','mean','scale','state_dict','seed')
    return {'members':[{k:m[k] for k in keep} for m in members],'feature_module':members[0]['feature_module'],
            'features':members[0]['features'],'selected_transition_costs':list(v8.COSTS),'min_controlled_frames':v8.FILTER}


def eval_game(models,game,cfg):
    obs,targets=retrain_v4.load_external(ROOT,game); truth=np.asarray(targets.label); seq=np.asarray(obs.sequence_id); dead=truth=='D'
    cells=[]
    for scenario in [s for s in cfg['scenarios'] if s['name'] in v8.EVAL_SCENARIOS]:
        held=v8.changed(obs,scenario,'step4',game); ball=np.array(held.ball_xy_m); ball[dead]=np.nan
        blank=replace(held,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
        for ball_name,base in (('held',held),('blank',blank)):
            faulty=tracking_faults.apply(base,seed_for('faults-step4',game,ball_name,scenario['name']))
            for players,work in (('faults',faulty),('all',base)):
                cache={}
                for name,model in models.items():
                    cells.append({'game':game,'scenario':scenario['name'],'ball':ball_name,'players':players,'model':name,
                                  **rb.score(truth,rb.predict(model,work,seq,cache),seq)})
    return cells


def logo():
    retrain_v4.changed=v9.changed
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261117,4)
    seeds=[41,42,43]; allcells=[]; sk={}
    for game in ('Sample_Game_1','Sample_Game_2'):
        other=[g for g in ('Sample_Game_1','Sample_Game_2') if g!=game]
        A=bundle(train_models(v8.DFL+other,seeds,f'A_not{game[-1]}'))
        B=bundle(train_models(v8.DFL+other+['Sample_Game_3'],seeds,f'B_not{game[-1]}'))
        cells=eval_game({'A':A,'B':B},game,cfg); allcells+=cells; sk[game]=(A,B)
        for cond in ('A','B'):
            m=lambda p,b:np.mean([c['spellF1'] for c in cells if c['model']==cond and c['players']==p and c['ball']==b and c['scenario']!='clean'])
            print(game,cond,'held+faults %.4f held+all %.4f blank+faults %.4f blank+all %.4f'%(m('faults','held'),m('all','held'),m('faults','blank'),m('all','blank')),flush=True)
    write_json(ROOT/'reports/bench/step4_logo.json',{'cells':allcells})
    # SkillCorner: ensembles of all six A models vs all six B models
    mem=lambda k:[m for g in sk for m in sk[g][k]['members']]
    models={c:{'members':mem(k),'feature_module':'v6','selected_transition_costs':list(v8.COSTS),'min_controlled_frames':v8.FILTER}
            for k,c in enumerate(('A','B'))}
    cells=rb.natural(models,module); write_json(ROOT/'reports/bench/step4_skillcorner.json',{'cells':cells})
    for c in ('A','B'):
        print('SkillCorner',c,'%.4f'%np.mean([x['spellF1'] for x in cells if x['model']==c]))


def final(n):
    retrain_v4.changed=v9.changed
    seeds=[9300+i for i in range(n)]
    members=train_models(ALL,seeds,'final')
    path=ROOT/'models/temporal_v10e.pt'
    if path.exists():
        raise FileExistsError('bundle exists')
    b=bundle(members)
    atomic_checkpoint(path,{'format_version':1,'kind':'pytorch_temporal_ensemble_v1',**b,
        'status':f'v10e: average of {n} networks trained on DFL x7 plus Metrica sample games 1-3 (weight 3); no held-out Metrica score exists'})
    print('bundle written',n)


if __name__=='__main__':
    stage=sys.argv[1]
    {'feats':feats,'logo':logo}.get(stage,lambda:final(int(sys.argv[2])))()
