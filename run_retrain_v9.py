"""v9: v8 plus player-tracking faults shaped like the competition tracking. External data only.

Adversarial check (run_adversarial_check.py) showed the largest input mismatch between the
competition tracking and every training scenario is the number of tracked players: a team is
short of 11 on about 36% of real frames, never in clean external data. The shipped v7 loses
0.03-0.07 spellF1 on Metrica's sample games when such faults are added (measured).

v9 applies fcb_external/tracking_faults.py on top of every ball-loss scenario during training
(the clean scenario stays clean).

Protocol (fixed before running):
  swap   train DFL x7 + sample game A (Metrica games sampled `weight` times), score game B,
         then the reverse; v7 and the matching v8 swap model are scored on the same cells,
         each with all players and with player faults.
  rule   adopt v9 only if, in BOTH directions, its masked mean with player faults beats v7's
         and v8's, and its masked mean with all players is not more than 0.01 below the
         better of them.
  final  refit on DFL x7 + both sample games.

    python run_retrain_v9.py swap <weight> | final <weight>
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import run_retrain_v8 as v8
from fcb_external import retrain_v4, temporal, tracking_faults
from fcb_external.feature_recovery_experiment import atomic_checkpoint, seed_for
from fcb_external.schema import digest_file, write_json

ROOT=Path(__file__).resolve().parent
_ball_only=v8.changed


def changed(obs,scenario,role,fold_name):
    work=_ball_only(obs,scenario,role,fold_name)
    if role=='train' and scenario['name']!='clean':
        work=tracking_faults.apply(work,seed_for('faults',fold_name,obs.match_id,scenario['name']))
    return work


def swap(weight):
    retrain_v4.changed=changed
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    load=lambda path:torch.load(path,map_location='cpu',weights_only=True)
    v7=load(ROOT/'models/temporal_v7.pt')
    scenarios=[s for s in cfg['scenarios'] if s['name'] in v8.EVAL_SCENARIOS]
    out_dir=ROOT/'models/temporal_v9'; out_dir.mkdir(parents=True,exist_ok=True); cells=[]
    for train_game,test_game in ((v8.GAMES[0],v8.GAMES[1]),(v8.GAMES[1],v8.GAMES[0])):
        fold=f'swap_{train_game[-1]}'; matches=v8.DFL+[train_game]; path=out_dir/f'{fold}_w{weight}.pt'
        if not path.exists():
            arrays,_=retrain_v4.training_arrays(ROOT,cfg,matches,fold)
            ck=retrain_v4.fit(v8.weighted(arrays,matches,weight),9100+10*int(train_game[-1])+weight,cfg['network'],cfg['training'],module)
            del arrays
            ck.update(selected_transition_costs=list(v8.COSTS),min_controlled_frames=v8.FILTER,fold=fold,metrica_weight=weight)
            atomic_checkpoint(path,ck)
        models={'v7':v7,'v8':load(ROOT/f'models/temporal_v8/{fold}_w{weight}.pt'),'v9':load(path)}
        obs,targets=retrain_v4.load_external(ROOT,test_game)
        truth=np.asarray(targets.label); seq=np.asarray(obs.sequence_id)
        for scenario in scenarios:
            full=_ball_only(obs,scenario,'validation',fold)
            faulty=tracking_faults.apply(full,seed_for('faults-eval',fold,test_game,scenario['name']))
            for variant,work in (('all_players',full),('player_faults',faulty)):
                for name,model in models.items():
                    spell,macro=v8.score(model,module,work,truth,seq)
                    cells.append({'train_game':train_game,'test_game':test_game,'scenario':scenario['name'],
                                  'players':variant,'model':name,'spellF1':spell,'macroF1':macro})
            print('scored',test_game,scenario['name'],flush=True)
    write_json(ROOT/f'reports/temporal_v9/swap_w{weight}.json',{'cells':cells,'weight':weight})
    for variant in ('player_faults','all_players'):
        print(f'\n{variant}: spellF1, masked mean per held-out game, then clean')
        for name in ('v7','v8','v9'):
            parts=[]
            for game in v8.GAMES:
                rows=[c for c in cells if c['model']==name and c['players']==variant and c['test_game']==game]
                parts.append('%s masked %.4f clean %.4f'%(game[-1],np.mean([c['spellF1'] for c in rows if c['scenario']!='clean']),
                                                           np.mean([c['spellF1'] for c in rows if c['scenario']=='clean'])))
            print(f'{name:<4}','   '.join(parts))
        print('per scenario (mean of both games):')
        for name in ('v7','v8','v9'):
            print(f'{name:<4}','  '.join('%s %.3f'%(s,np.mean([c['spellF1'] for c in cells if c['model']==name and c['players']==variant and c['scenario']==s])) for s in v8.EVAL_SCENARIOS))


def final(weight):
    retrain_v4.changed=changed
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    path=ROOT/'models/temporal_v9.pt'; submission=ROOT/'configs/submission_temporal_v9.json'
    if path.exists() or submission.exists():
        raise FileExistsError('Final v9 model or submission config already exists')
    matches=v8.DFL+v8.GAMES
    arrays,inputs=retrain_v4.training_arrays(ROOT,cfg,matches,'final')
    ck=retrain_v4.fit(v8.weighted(arrays,matches,weight),9105,cfg['network'],cfg['training'],module)
    ck.update(selected_transition_costs=list(v8.COSTS),min_controlled_frames=v8.FILTER,training_inputs=inputs,metrica_weight=weight,
              status='v9: DFL x7 plus Metrica sample games 1 and 2, with target-like ball loss and player faults; no held-out score exists for this exact checkpoint')
    atomic_checkpoint(path,ck)
    sub=json.loads((ROOT/'configs/submission_temporal_v7.json').read_text())
    sub.update(model='../models/temporal_v9.pt',model_sha256=digest_file(path))
    write_json(submission,sub)
    print('Final v9 model and submission config written.')


if __name__=='__main__':
    (swap if sys.argv[1]=='swap' else final)(int(sys.argv[2]))
