"""Adversarial validation without labels: do the model's inputs on the competition tracking
look like its inputs on the simulated external matches?

For each of the 78 features, compare its distribution on the competition match's graded frames
with its distribution on external matches under each ball-loss scenario (two-sample KS
statistic on random frame samples; 0 = identical, 1 = disjoint). Features that differ a lot
are inputs the network has never seen in that range. Only per-feature summary statistics are
printed; no labels are read and no tracking values are shown.

    python run_adversarial_check.py --data-dir <bundle>
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from fcb_external import atd, team_features_v6
from fcb_external.retrain_v4 import load_external
from fcb_external.schema import write_json
from fcb_external.submission import timeline_blocks
from run_dead_ball_analysis import changed

ROOT=Path(__file__).resolve().parent
SOURCES=('J03WMX','J03WR9','Sample_Game_1')
SCENARIOS=('clean','irregular40','slow40','mixed_stress','target_like40')
SAMPLE=40000


def ks(a,b):
    grid=np.sort(np.concatenate([a,b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a),grid,side='right')/len(a)-np.searchsorted(np.sort(b),grid,side='right')/len(b))))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args(); rng=np.random.default_rng(20261116)
    sub=json.loads((ROOT/'configs/submission_temporal_v7.json').read_text())
    folder=args.data_dir/'tactical'; raw=folder/(sub['stem']+'_atd.txt'); meta=folder/(sub['stem']+'_atd.xml')
    teams=[t.get('id') for t in ET.parse(meta).getroot().iter('Team')]
    obs=atd.convert(raw,meta,sub['match_id'],teams[0],teams[1],*sub['pitch_m'],timeline_blocks(sub['n_frames'],sub['active_blocks']))
    graded=np.flatnonzero(np.asarray(obs.period)>0)
    target=team_features_v6.extract(obs)[rng.choice(graded,SAMPLE,replace=False)]
    cfg=json.loads((ROOT/'configs/retrain_v7.json').read_text())
    names=list(team_features_v6.NAMES); table={}
    for scenario in [s for s in cfg['scenarios'] if s['name'] in SCENARIOS]:
        parts=[]
        for match in SOURCES:
            source,_=load_external(ROOT,match)
            x=team_features_v6.extract(changed(source,scenario,'adversarial','check'))
            parts.append(x[rng.choice(len(x),SAMPLE//len(SOURCES),replace=False)])
        external=np.concatenate(parts)
        table[scenario['name']]={n:{'ks':ks(target[:,i],external[:,i]),'target_mean':float(target[:,i].mean()),
                                    'external_mean':float(external[:,i].mean()),'target_p90':float(np.percentile(target[:,i],90)),
                                    'external_p90':float(np.percentile(external[:,i],90))} for i,n in enumerate(names)}
        print('compared',scenario['name'],'mean KS %.3f'%np.mean([v['ks'] for v in table[scenario['name']].values()]),flush=True)
    write_json(ROOT/'reports/adversarial_check/results.json',table)
    best=min(table,key=lambda s:np.mean([v['ks'] for v in table[s].values()]))
    print('\nclosest scenario overall:',best)
    print(f'{"feature":<28}{"KS":>6}{"target mean":>13}{"external mean":>15}{"target p90":>12}{"external p90":>14}')
    for name,row in sorted(table[best].items(),key=lambda kv:-kv[1]['ks'])[:28]:
        print(f'{name:<28}{row["ks"]:>6.2f}{row["target_mean"]:>13.3f}{row["external_mean"]:>15.3f}{row["target_p90"]:>12.3f}{row["external_p90"]:>14.3f}')


if __name__=='__main__':
    main()
