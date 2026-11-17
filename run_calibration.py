"""Label-free decoder calibration.

External part (always): on Metrica's public sample matches with the ball masked, score a grid
of decoder settings for the single v4 model and for the average of the five v4 models
(four cross-validation folds plus the final refit). Reports spellF1 and predicted possessions.

Target part (only with --data-dir): for the same grid, report how many possessions and how much
dead ball each setting PREDICTS on the competition tracking. No competition answers are read;
only statistics of the system's own output are printed, never labels or tracking values.

    python run_calibration.py [--data-dir <bundle>]
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from fcb_external import atd, metrica_sample, temporal, team_features_v4
from fcb_external.feature_recovery_experiment import infer, seed_for
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells, runs
from fcb_external.retrain_v4 import decode_many
from fcb_external.robustness import corrupt
from fcb_external.schema import write_json
from fcb_external.submission import timeline_blocks

ROOT=Path(__file__).resolve().parent
OWNER=(5,10,20,40,80,160); DEAD=(10,30); FILTERS=(25,50)
PAIRS=[(float(o),float(d)) for o in OWNER for d in DEAD]
NAMES=np.array(['H','A','D'])


def checkpoints():
    load=lambda path:torch.load(path,map_location='cpu',weights_only=True)
    final=load(ROOT/'models/temporal_v4.pt')
    folds=[load(path) for path in sorted((ROOT/'models/temporal_v4').glob('[A-D].pt'))]
    return {'single':[final],'ensemble':[final]+folds}


def probabilities(models,obs):
    return np.mean([infer(model,obs,team_features_v4) for model in models],axis=0)


def decoded(p,seq):
    for pair,row in zip(PAIRS,decode_many(p,seq,PAIRS)):
        for frames in FILTERS:
            yield (pair[0],pair[1],frames),absorb_short_spells(NAMES[row],seq,frames)


def output_stats(label,seq,graded):
    lengths=np.array([b-a for a,b in runs(label,seq) if label[a] in ('H','A') and graded[a]])
    live=graded&np.isin(label,['H','A'])
    return {'possessions':int(len(lengths)),'median_s':float(np.median(lengths))/25,
            'dead_share':float(np.mean(label[graded]=='D')),
            'possessions_per_live_minute':float(len(lengths)/(live.sum()/1500))}


def external(models,scenarios):
    rows={}
    for name,(obs,truth) in metrica_sample.load_all(ROOT/'external-data/metrica_sample').items():
        seq=np.asarray(obs.sequence_id); graded=np.ones(len(seq),bool)
        truth_stats=output_stats(truth,seq,graded)
        for scenario in scenarios:
            work=corrupt(obs,{**scenario,'seed':seed_for('calibration','metrica',name,scenario['name'])})
            for kind,group in models.items():
                for key,label in decoded(probabilities(group,work),seq):
                    cell=rows.setdefault((kind,)+key,{'spellF1':[],'strict':[],'possessions':[],'per_live_minute':[],'dead_share':[]})
                    cell['spellF1'].append(evaluate_labels(truth,label,seq)['metrics']['spell_f1_mean'])
                    cell['strict'].append(evaluate_labels(absorb_short_spells(truth,seq,50),label,seq)['metrics']['spell_f1_mean'])
                    stats=output_stats(label,seq,graded)
                    cell['possessions'].append(stats['possessions']); cell['dead_share'].append(stats['dead_share'])
                    cell['per_live_minute'].append(stats['possessions_per_live_minute'])
            print('external',name,scenario['name'],truth_stats,flush=True)
    return {key:{k:float(np.mean(v)) for k,v in cell.items()} for key,cell in rows.items()}


def target(models,data_dir):
    cfg=json.loads((ROOT/'configs/submission_final.json').read_text())
    root=Path(data_dir)/'tactical'; raw=root/(cfg['stem']+'_atd.txt'); meta=root/(cfg['stem']+'_atd.xml')
    teams=[team.get('id') for team in ET.parse(meta).getroot().iter('Team')]
    blocks=timeline_blocks(cfg['n_frames'],cfg['active_blocks'])
    obs=atd.convert(raw,meta,cfg['match_id'],teams[0],teams[1],*cfg['pitch_m'],blocks)
    seq=np.asarray(obs.sequence_id); graded=np.asarray(obs.period)>0
    rows={}
    for kind,group in models.items():
        for key,label in decoded(probabilities(group,obs),seq):
            rows[(kind,)+key]=output_stats(label,seq,graded)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args()
    temporal.configure(20261115,4)
    cfg=json.loads((ROOT/'configs/retrain_v4.json').read_text())
    scenarios=[s for s in cfg['scenarios'] if s['name'] in ('irregular40','mixed_stress')]
    models=checkpoints()
    ext=external(models,scenarios)
    tgt=target(models,args.data_dir) if args.data_dir else {}
    write_json(ROOT/'reports/calibration/results.json',
               {'external':{'/'.join(map(str,k)):v for k,v in ext.items()},
                'target_output_statistics':{'/'.join(map(str,k)):v for k,v in tgt.items()}})
    print(f'{"model":<9}{"owner":>6}{"dead":>5}{"filt":>5} | {"extF1":>6}{"strict":>7}{"ext/min":>8}{"extDead":>8} | {"tgtPoss":>8}{"tgt/min":>8}{"tgtDead":>8}')
    for key,row in ext.items():
        t=tgt.get(key)
        tail=f'{t["possessions"]:>8}{t["possessions_per_live_minute"]:>8.2f}{t["dead_share"]:>8.3f}' if t else ''
        print(f'{key[0]:<9}{key[1]:>6.0f}{key[2]:>5.0f}{key[3]:>5} | {row["spellF1"]:>6.3f}{row["strict"]:>7.3f}{row["per_live_minute"]:>8.2f}{row["dead_share"]:>8.3f} | {tail}')


if __name__=='__main__':
    main()
