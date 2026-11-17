"""Calibrate dead-ball edge offsets on Metrica's public sample matches (external data only).

The v7 pipeline's predicted stoppages start late and end early relative to the organiser's
own corrected tracking on these other matches. This scores a small grid of fixed offsets.

    python run_dead_edge_calibration.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, state_decoder, temporal
from fcb_external.feature_recovery_experiment import infer
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells, shift_dead_edges
from fcb_external.retrain_v4 import MODULES
from fcb_external.schema import write_json
from run_dead_ball_analysis import changed

ROOT=Path(__file__).resolve().parent
SCENARIOS=('clean','irregular40','crowded40','mixed_stress','slow40','target_like40')
STARTS=(0,5,10,15,20,25,35); ENDS=(0,5,10,15)


def main():
    cfg=json.loads((ROOT/'configs/retrain_v7.json').read_text())
    temporal.configure(20261116,4)
    model=torch.load(ROOT/'models/temporal_v7.pt',map_location='cpu',weights_only=True)
    module=MODULES[model.get('feature_module','v4')]; owner,dead=model['selected_transition_costs']
    cells=[]
    for name,(obs,truth) in metrica_sample.load_all(ROOT/'external-data/metrica_sample').items():
        seq=np.asarray(obs.sequence_id)
        for scenario in [s for s in cfg['scenarios'] if s['name'] in SCENARIOS]:
            work=changed(obs,scenario,'validation','metrica')
            base=absorb_short_spells(state_decoder.decode(infer(model,work,module),seq,owner,dead),seq,25)
            for s in STARTS:
                for e in ENDS:
                    m=evaluate_labels(truth,shift_dead_edges(base,seq,s,e),seq)['metrics']
                    cells.append({'match':name,'scenario':scenario['name'],'start_earlier':s,'end_later':e,
                                  'spellF1':float(m['spell_f1_mean']),'macroF1':float(m['macro_f1'])})
            print('scored',name,scenario['name'],flush=True)
    write_json(ROOT/'reports/dead_ball_analysis/edge_calibration.json',{'cells':cells})
    matches=sorted({c['match'] for c in cells})
    def mean(s,e,match=None,masked=True):
        rows=[c['spellF1'] for c in cells if c['start_earlier']==s and c['end_later']==e and (match is None or c['match']==match) and ((c['scenario']!='clean')==masked)]
        return float(np.mean(rows))
    print(f'{"start":>6}{"end":>5}{"masked":>9}{"clean":>8}  '+'  '.join(matches)+'   worst-cell delta')
    for s in STARTS:
        for e in ENDS:
            deltas=[c['spellF1']-next(b['spellF1'] for b in cells if b['match']==c['match'] and b['scenario']==c['scenario'] and b['start_earlier']==0 and b['end_later']==0)
                    for c in cells if c['start_earlier']==s and c['end_later']==e]
            print(f'{s:>6}{e:>5}{mean(s,e):>9.4f}{mean(s,e,masked=False):>8.4f}  '+'  '.join(f'{mean(s,e,m):.4f}' for m in matches)+f'   {min(deltas):+.4f}')


if __name__=='__main__':
    main()
