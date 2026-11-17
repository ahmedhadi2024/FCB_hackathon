"""Where does dead-ball detection fail? External matches only (Metrica public samples).

For the shipped v7 pipeline under several ball-loss scenarios, report per-class dead-ball
precision and recall, how many true stoppages are missed outright (by duration), and how
early or late the detected ones start and end. No competition data is read.

    python run_dead_ball_analysis.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, state_decoder, temporal, target_like
from fcb_external.feature_recovery_experiment import infer, seed_for
from fcb_external.possession_convention import absorb_short_spells, runs
from fcb_external.retrain_v4 import MODULES, changed as base_changed
from fcb_external.schema import write_json

ROOT=Path(__file__).resolve().parent
SCENARIOS=('clean','irregular40','slow40','target_like40')
BINS=((0,5),(5,10),(10,20),(20,40),(40,1e9))


def changed(obs,scenario,role,fold):
    if scenario.get('kind')=='target_like':
        return target_like.mask(obs,scenario['missing_fraction'],scenario['slow_weight'],scenario['line_weight'],
                                seed_for(role,fold,obs.match_id,scenario['name']))
    return base_changed(obs,scenario,role,fold)


def dead_segments(label,seq):
    return [(a,b) for a,b in runs(label,seq) if label[a]=='D']


def analyse(truth,pred,seq):
    t=truth=='D'; p=pred=='D'
    out={'precision':float((t&p).sum()/max(p.sum(),1)),'recall':float((t&p).sum()/max(t.sum(),1)),
         'true_share':float(t.mean()),'pred_share':float(p.mean())}
    predicted=dead_segments(pred,seq); starts=np.array([a for a,_ in predicted]); ends=np.array([b for _,b in predicted])
    missed={f'{lo}-{hi if hi<1e8 else "inf"}s':[0,0] for lo,hi in BINS}; start_err=[]; end_err=[]; missed_frames=0; edge_frames=0
    for a,b in dead_segments(truth,seq):
        seconds=(b-a)/25; key=next(f'{lo}-{hi if hi<1e8 else "inf"}s' for lo,hi in BINS if lo<=seconds<hi)
        missed[key][1]+=1
        overlap=np.flatnonzero((starts<b)&(ends>a)) if len(starts) else []
        if not len(overlap):
            missed[key][0]+=1; missed_frames+=b-a; continue
        first=starts[overlap].min(); last=ends[overlap].max()
        start_err.append((first-a)/25); end_err.append((last-b)/25)
        edge_frames+=max(0,first-a)+max(0,b-last)
    out.update(missed_by_duration={k:f'{m}/{n}' for k,(m,n) in missed.items()},
               share_of_missed_dead_frames_in_missed_stoppages=float(missed_frames/max((t&~p).sum(),1)),
               share_of_missed_dead_frames_at_edges=float(edge_frames/max((t&~p).sum(),1)),
               start_late_s={'median':float(np.median(start_err)),'p25':float(np.percentile(start_err,25)),'p75':float(np.percentile(start_err,75))},
               end_late_s={'median':float(np.median(end_err)),'p25':float(np.percentile(end_err,25)),'p75':float(np.percentile(end_err,75))},
               true_stoppages=len(dead_segments(truth,seq)),pred_stoppages=len(predicted))
    return out


def main():
    cfg=json.loads((ROOT/'configs/retrain_v7.json').read_text())
    temporal.configure(20261116,4)
    model=torch.load(ROOT/'models/temporal_v7.pt',map_location='cpu',weights_only=True)
    module=MODULES[model.get('feature_module','v4')]; owner,dead=model['selected_transition_costs']
    report={}
    for name,(obs,truth) in metrica_sample.load_all(ROOT/'external-data/metrica_sample').items():
        seq=np.asarray(obs.sequence_id)
        for scenario in [s for s in cfg['scenarios'] if s['name'] in SCENARIOS]:
            work=changed(obs,scenario,'validation','metrica')
            pred=absorb_short_spells(state_decoder.decode(infer(model,work,module),seq,owner,dead),seq,25)
            report[f'{name}/{scenario["name"]}']=row=analyse(truth,pred,seq)
            print(name,scenario['name'],json.dumps(row),flush=True)
    write_json(ROOT/'reports/dead_ball_analysis/v7.json',report)


if __name__=='__main__':
    main()
