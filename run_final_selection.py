"""Final selection among v7, v8 and v9 on Metrica's sample games (external data only).

Every candidate is scored on the sample game it did not train on, in both directions, under
six ball-loss scenarios and four input conditions:
  ball held / blank      ball kept at its last position through stoppages, or blank during them
  all players / faults   every player tracked, or the measured missing-player profile

Rule (fixed before the candidates were compared on all conditions): a candidate qualifies only
if, in BOTH directions, (a) on held+faults, the condition closest to the competition tracking,
its masked mean beats v7; and (b) on each of the other three conditions it is not more than
0.01 below v7. Among qualifiers the highest held+faults masked mean wins.

    python run_final_selection.py
"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import run_retrain_v8 as v8
from fcb_external import retrain_v4, state_decoder, temporal, tracking_faults
from fcb_external.feature_recovery_experiment import seed_for
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.schema import write_json

ROOT=Path(__file__).resolve().parent
CONDITIONS=[('held','faults'),('held','all'),('blank','faults'),('blank','all')]


def probabilities(model,x,seq,module):
    net=temporal.TemporalNet(len(module.NAMES),model['network']['channels'],model['network']['dilations'])
    net.load_state_dict(model['state_dict'])
    mean=model['mean'].numpy(); scale=model['scale'].numpy()
    a=temporal.probabilities(net,(x-mean)/scale,seq); b=temporal.probabilities(net,(module.swap(x)-mean)/scale,seq)
    return (a+b[:,[1,0,2]])*.5


def main():
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    load=lambda path:torch.load(path,map_location='cpu',weights_only=True)
    scenarios=[s for s in cfg['scenarios'] if s['name'] in v8.EVAL_SCENARIOS]; cells=[]
    for train_game,test_game in ((v8.GAMES[0],v8.GAMES[1]),(v8.GAMES[1],v8.GAMES[0])):
        fold=f'swap_{train_game[-1]}'
        models={'v7':load(ROOT/'models/temporal_v7.pt')}
        for version in ('v8','v9'):
            for w in (1,3):
                path=ROOT/f'models/temporal_{version}/{fold}_w{w}.pt'
                if path.exists():
                    models[f'{version}_w{w}']=load(path)
        obs,targets=retrain_v4.load_external(ROOT,test_game)
        truth=np.asarray(targets.label); seq=np.asarray(obs.sequence_id); dead=truth=='D'
        for scenario in scenarios:
            held=v8.changed(obs,scenario,'validation',fold)
            ball=np.array(held.ball_xy_m); ball[dead]=np.nan
            blank=replace(held,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
            for ball_name,base in (('held',held),('blank',blank)):
                faulty=tracking_faults.apply(base,seed_for('faults-eval',fold,test_game,scenario['name']))
                for players,work in (('faults',faulty),('all',base)):
                    x=module.extract(work)
                    for name,model in models.items():
                        d=state_decoder.decode(probabilities(model,x,seq,module),seq,*v8.COSTS)
                        m=evaluate_labels(truth,absorb_short_spells(d,seq,v8.FILTER),seq)['metrics']
                        cells.append({'test_game':test_game,'scenario':scenario['name'],'ball':ball_name,'players':players,
                                      'model':name,'spellF1':float(m['spell_f1_mean']),'macroF1':float(m['macro_f1'])})
            print('scored',test_game,scenario['name'],flush=True)
    names=list(dict.fromkeys(c['model'] for c in cells))
    def masked(model,cond,game):
        return float(np.mean([c['spellF1'] for c in cells if c['model']==model and (c['ball'],c['players'])==cond
                              and c['test_game']==game and c['scenario']!='clean']))
    table={n:{f'{b}+{p}':[masked(n,(b,p),g) for g in v8.GAMES] for b,p in CONDITIONS} for n in names}
    qualifies={}
    for n in names:
        if n=='v7':
            continue
        main_ok=all(table[n]['held+faults'][i]>table['v7']['held+faults'][i] for i in (0,1))
        other_ok=all(table[n][f'{b}+{p}'][i]>=table['v7'][f'{b}+{p}'][i]-0.01 for b,p in CONDITIONS[1:] for i in (0,1))
        qualifies[n]=bool(main_ok and other_ok)
    passed=[n for n,ok in qualifies.items() if ok]
    winner=max(passed,key=lambda n:np.mean(table[n]['held+faults'])) if passed else 'v7'
    write_json(ROOT/'reports/final_selection/results.json',{'cells':cells,'masked_means':table,'qualifies':qualifies,'winner':winner})
    print(f'\n{"model":<8}'+''.join(f'{b+"+"+p:>22}' for b,p in CONDITIONS)+'   qualifies')
    for n in names:
        print(f'{n:<8}'+''.join('%22s'%('%.4f / %.4f'%tuple(table[n][f"{b}+{p}"])) for b,p in CONDITIONS)+f'   {qualifies.get(n,"-")}')
    print('\nwinner:',winner)


if __name__=='__main__':
    main()
