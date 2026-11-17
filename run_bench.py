"""Stage 0 test bench: matches no shipped model has trained on. External data only.

Two tracks, both scored with the unmodified official metric on the provider's labels as
derived (no merging of short possessions):

  metrica3   Metrica public sample game 3, the organiser's own tracking and event data. Six
             ball-loss scenarios x {ball held, ball blank during stoppages} x {all players,
             measured player faults}.
  natural    SkillCorner open data: real broadcast tracking with the provider's own ball
             faults (ball used only where detected; player positions as supplied, including
             extrapolated ones). No simulated fault is added. Labels are the provider's
             phases of play, so this is a different labelling convention and a far harsher
             camera than the competition's tactical view: a cross-provider robustness test.

    python run_bench.py [model names ...]      default: temporal_v7 temporal_v9
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import run_retrain_v8 as v8
from fcb_external import metrica_sample, retrain_v4, skillcorner, state_decoder, temporal, tracking_faults
from fcb_external.feature_recovery_experiment import seed_for
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.schema import write_json

ROOT=Path(__file__).resolve().parent
SC=ROOT/'external-data/skillcorner'


def probabilities(model,x,seq,module):
    net=temporal.TemporalNet(x.shape[1],model['network']['channels'],model['network']['dilations'])
    net.load_state_dict(model['state_dict'])
    mean=model['mean'].numpy(); scale=model['scale'].numpy()
    a=temporal.probabilities(net,(x-mean)/scale,seq); b=temporal.probabilities(net,(module.swap(x)-mean)/scale,seq)
    return (a+b[:,[1,0,2]])*.5


def predict(model,work,seq,cache):
    """Shipped pipeline for a checkpoint, with features from the checkpoint's own feature module."""
    key=model.get('feature_module','v4'); module=retrain_v4.MODULES[key]
    if key not in cache:
        cache[key]=module.extract(work)
    x=cache[key]
    owner,dead=model.get('selected_transition_costs',v8.COSTS)
    members=model.get('members') or [model]      # an ensemble bundle averages its members
    p=np.mean([probabilities(member,x,seq,module) for member in members],axis=0)
    decoded=state_decoder.decode(p/p.sum(axis=1,keepdims=True),seq,owner,dead)
    return absorb_short_spells(decoded,seq,int(model.get('min_controlled_frames',v8.FILTER) or v8.FILTER))


def score(truth,pred,seq):
    r=evaluate_labels(truth,pred,seq); m=r['metrics']
    return {'spellF1':float(m['spell_f1_mean']),'macroF1':float(m['macro_f1']),'pred_spells':int(m['n_pred_spells']),
            'gt_spells':int(m['n_gt_spells']),'dead_recall':float(r['per_class']['D']['recall']),
            'dead_precision':float(r['per_class']['D']['precision'])}


def metrica3(models,module,cfg):
    obs,events=metrica_sample.load_epts_game(ROOT/'external-data/metrica_sample')
    truth=metrica_sample.event_labels(obs,events); seq=np.asarray(obs.sequence_id); dead=truth=='D'
    held_obs=v8.hold_ball(obs); cells=[]
    for scenario in [s for s in cfg['scenarios'] if s['name'] in v8.EVAL_SCENARIOS]:
        held=v8.changed(held_obs,scenario,'bench','metrica3')
        ball=np.array(held.ball_xy_m); ball[dead]=np.nan
        blank=replace(held,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()
        for ball_name,base in (('held',held),('blank',blank)):
            faulty=tracking_faults.apply(base,seed_for('faults-bench','metrica3',ball_name,scenario['name']))
            for players,work in (('faults',faulty),('all',base)):
                cache={}
                for name,model in models.items():
                    cells.append({'track':'metrica3','scenario':scenario['name'],'ball':ball_name,'players':players,
                                  'model':name,**score(truth,predict(model,work,seq,cache),seq)})
        print('metrica3',scenario['name'],flush=True)
    return cells


def natural(models,module):
    cells=[]
    for meta in json.loads((SC/'matches.json').read_text()):
        mid=meta['id']
        if not all((SC/name).exists() for name in skillcorner.files(mid)):
            continue
        obs,truth=skillcorner.load_match(SC,mid); seq=np.asarray(obs.sequence_id); cache={}
        for name,model in models.items():
            cells.append({'track':'natural','match':str(mid),'model':name,**score(truth,predict(model,obs,seq,cache),seq)})
        print('natural',mid,flush=True)
    return cells


def main(names):
    cfg=v8.config(); module=retrain_v4.module_for(cfg); temporal.configure(20261116,4)
    models={n:torch.load(ROOT/'models'/f'{n}.pt',map_location='cpu',weights_only=True) for n in names}
    cells=metrica3(models,module,cfg)+natural(models,module)
    write_json(ROOT/'reports/bench'/('_'.join(names)+'.json'),{'cells':cells})
    mean=lambda rows,key='spellF1':float(np.mean([c[key] for c in rows])) if rows else float('nan')
    print(f'\n{"model":<14}{"M3 held+faults":>16}{"M3 held+all":>13}{"M3 blank+faults":>17}{"M3 blank+all":>14}{"M3 clean h+f":>14}{"natural":>9}{"nat pred/true":>15}{"nat D rec":>10}')
    for n in names:
        mine=[c for c in cells if c['model']==n]; m3=[c for c in mine if c['track']=='metrica3']; nat=[c for c in mine if c['track']=='natural']
        def cond(b,p,clean=False):
            return mean([c for c in m3 if c['ball']==b and c['players']==p and (c['scenario']=='clean')==clean])
        print(f'{n:<14}{cond("held","faults"):>16.4f}{cond("held","all"):>13.4f}{cond("blank","faults"):>17.4f}{cond("blank","all"):>14.4f}'
              f'{cond("held","faults",True):>14.4f}{mean(nat):>9.4f}{("%d/%d"%(mean(nat,"pred_spells"),mean(nat,"gt_spells"))):>15}{mean(nat,"dead_recall"):>10.2f}')
    print('natural matches:',len({c['match'] for c in cells if c['track']=='natural'}))


if __name__=='__main__':
    main(sys.argv[1:] or ['temporal_v7','temporal_v9'])
