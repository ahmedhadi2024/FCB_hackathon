"""v8: add Metrica's public sample matches to the training data. External data only.

Why: the seven DFL matches are labelled under another provider's convention. Metrica's two
public sample games are other matches of the organiser's own corrected tracking, with labels
derived from its event data (fcb_external/metrica_sample.py).

Preparation. In the samples the corrected ball is blank exactly while the ball is dead, which
no automatic tracking reproduces and which would teach "ball missing = dead". The ball is
therefore held at its last position through each blank run (as the DFL data does) before any
ball-loss scenario is applied.

Protocol (fixed before running):
  swap   train DFL x7 + sample game A, score sample game B; then the reverse. The shipped v7
         (never trained on either game) is scored on the same cells. Two variants: Metrica
         games counted once, or three times, in the sampling.
  rule   adopt a variant only if its masked mean beats v7 in BOTH directions with the held
         ball, and does not fall more than 0.01 below v7 with the blank ball restored.
  final  refit on DFL x7 + both sample games with the adopted variant.

    python run_retrain_v8.py prepare | swap | final <weight>
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, retrain_v4, state_decoder, target_like, temporal
from fcb_external.feature_recovery_experiment import atomic_checkpoint, infer, seed_for
from fcb_external.features import sequence_spans
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.schema import Targets, digest_file, write_json

ROOT=Path(__file__).resolve().parent
DFL=['J03WMX','J03WOH','J03WOY','J03WPY','J03WQQ','J03WN1','J03WR9']
GAMES=['Sample_Game_1','Sample_Game_2']
TRAIN_SCENARIOS=['clean','irregular40','crowded40','mixed_stress','slow40','slow60','target_like40']
EVAL_SCENARIOS=['clean','irregular40','crowded40','mixed_stress','slow40','target_like40']
COSTS=(5.0,30.0); FILTER=25; WEIGHTS=(1,3)
_base_changed=retrain_v4.changed


def changed(obs,scenario,role,fold_name):
    if scenario.get('kind')=='target_like':
        return target_like.mask(obs,scenario['missing_fraction'],scenario['slow_weight'],scenario['line_weight'],
                                seed_for(role,fold_name,obs.match_id,scenario['name']))
    return _base_changed(obs,scenario,role,fold_name)


retrain_v4.changed=changed


def hold_ball(obs):
    """Carry the last ball position through blank runs (first position backwards at a start)."""
    ball=np.array(obs.ball_xy_m,dtype=np.float32)
    for lo,hi in sequence_spans(obs.sequence_id):
        seen=np.flatnonzero(np.isfinite(ball[lo:hi]).all(axis=1))
        if not len(seen):
            continue
        index=np.searchsorted(seen,np.arange(hi-lo),side='right')-1
        ball[lo:hi]=ball[lo:hi][seen[np.maximum(index,0)]]
    return replace(obs,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()


def prepare():
    for name,(obs,label) in metrica_sample.load_all(ROOT/'external-data/metrica_sample').items():
        folder=ROOT/'artifacts'/name
        if (folder/'observations/metadata.json').exists():
            print('Already prepared:',name); continue
        held=hold_ball(obs)
        held.save(folder/'observations'); Targets(label,label!='X').save(folder/'targets',held)
        print('Prepared',name,'frames',len(label),{k:int((label==k).sum()) for k in 'HADX'},flush=True)


def config():
    cfg=json.loads((ROOT/'configs/retrain_v7.json').read_text())
    cfg['training_scenarios']=TRAIN_SCENARIOS
    return cfg


def weighted(arrays,matches,weight):
    """Repeat the references of the Metrica matches so they are sampled `weight` times as often."""
    per=len(TRAIN_SCENARIOS); out=list(arrays)
    for i,match in enumerate(matches):
        if match in GAMES:
            out+=arrays[i*per:(i+1)*per]*(weight-1)
    return out


def score(model,module,work,truth,seq):
    decoded=state_decoder.decode(infer(model,work,module),seq,*COSTS)
    m=evaluate_labels(truth,absorb_short_spells(decoded,seq,FILTER),seq)['metrics']
    return float(m['spell_f1_mean']),float(m['macro_f1'])


def swap():
    cfg=config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    v7=torch.load(ROOT/'models/temporal_v7.pt',map_location='cpu',weights_only=True)
    scenarios=[s for s in cfg['scenarios'] if s['name'] in EVAL_SCENARIOS]
    out_dir=ROOT/'models/temporal_v8'; out_dir.mkdir(parents=True,exist_ok=True)
    cells=[]
    for train_game,test_game in ((GAMES[0],GAMES[1]),(GAMES[1],GAMES[0])):
        fold=f'swap_{train_game[-1]}'; matches=DFL+[train_game]
        models={'v7':v7}; missing=[w for w in WEIGHTS if not (out_dir/f'{fold}_w{w}.pt').exists()]
        if missing:
            arrays,_=retrain_v4.training_arrays(ROOT,cfg,matches,fold)
            for w in missing:
                ck=retrain_v4.fit(weighted(arrays,matches,w),8100+10*int(train_game[-1])+w,cfg['network'],cfg['training'],module)
                ck.update(selected_transition_costs=list(COSTS),min_controlled_frames=FILTER,fold=fold,metrica_weight=w)
                atomic_checkpoint(out_dir/f'{fold}_w{w}.pt',ck)
            del arrays
        for w in WEIGHTS:
            models[f'v8_w{w}']=torch.load(out_dir/f'{fold}_w{w}.pt',map_location='cpu',weights_only=True)
        obs,targets=retrain_v4.load_external(ROOT,test_game)
        truth=np.asarray(targets.label); seq=np.asarray(obs.sequence_id); dead=truth=='D'
        for scenario in scenarios:
            held=changed(obs,scenario,'validation',fold)
            ball=np.array(held.ball_xy_m); ball[dead]=np.nan
            blank=replace(held,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
            for variant,work in (('held',held),('blank',blank)):
                for name,model in models.items():
                    spell,macro=score(model,module,work,truth,seq)
                    cells.append({'train_game':train_game,'test_game':test_game,'scenario':scenario['name'],
                                  'ball':variant,'model':name,'spellF1':spell,'macroF1':macro})
            print('scored',test_game,scenario['name'],flush=True)
    write_json(ROOT/'reports/temporal_v8/swap.json',{'cells':cells,'costs':COSTS,'filter':FILTER})
    names=['v7']+[f'v8_w{w}' for w in WEIGHTS]
    for variant in ('held','blank'):
        print(f'\nball {variant}: spellF1, masked mean per held-out game, then clean')
        for name in names:
            parts=[]
            for game in GAMES:
                rows=[c for c in cells if c['model']==name and c['ball']==variant and c['test_game']==game]
                parts.append('%s masked %.4f clean %.4f'%(game[-1],np.mean([c['spellF1'] for c in rows if c['scenario']!='clean']),
                                                           np.mean([c['spellF1'] for c in rows if c['scenario']=='clean'])))
            print(f'{name:<7}','   '.join(parts))
        print('per scenario (mean of both games):')
        for name in names:
            print(f'{name:<7}','  '.join('%s %.3f'%(s,np.mean([c['spellF1'] for c in cells if c['model']==name and c['ball']==variant and c['scenario']==s])) for s in EVAL_SCENARIOS))


def final(weight):
    cfg=config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    path=ROOT/'models/temporal_v8.pt'; submission=ROOT/'configs/submission_temporal_v8.json'
    if path.exists() or submission.exists():
        raise FileExistsError('Final v8 model or submission config already exists')
    matches=DFL+GAMES
    arrays,inputs=retrain_v4.training_arrays(ROOT,cfg,matches,'final')
    ck=retrain_v4.fit(weighted(arrays,matches,weight),8105,cfg['network'],cfg['training'],module)
    ck.update(selected_transition_costs=list(COSTS),min_controlled_frames=FILTER,training_inputs=inputs,metrica_weight=weight,
              status='v8: DFL x7 plus Metrica public sample games 1 and 2; no held-out score exists for this exact checkpoint')
    atomic_checkpoint(path,ck)
    sub=json.loads((ROOT/'configs/submission_temporal_v7.json').read_text())
    sub.update(model='../models/temporal_v8.pt',model_sha256=digest_file(path))
    write_json(submission,sub)
    print('Final v8 model and submission config written.')


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare': prepare()
    elif stage=='swap': swap()
    else: final(int(sys.argv[2]))
