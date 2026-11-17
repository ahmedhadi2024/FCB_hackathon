"""Frozen external-development possession-duration experiment. No competition files accessed."""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import temporal
from fcb_external.experiment import load_match
from fcb_external.temporal_experiment import infer
from fcb_external.robustness import corrupt
from fcb_external.possession_duration import decode as duration_decode
from fcb_external.error_analysis import analyze
from fcb_external.schema import write_json,digest_file
from fcb_external.bounded_experiment import frozen_scenarios,capture_provenance,verify_provenance


def run():
    root=Path(__file__).parent
    output=root/'reports/possession_duration_v2.json'
    if output.exists():raise FileExistsError('Experiment already recorded')
    runner=Path(__file__).name
    model_path='models/temporal_v2.pt'
    match='J03WQQ'
    provenance=capture_provenance(root,runner,model_path,match)
    protocol=json.loads((root/'configs/experiment_v1.json').read_text())
    scenarios=frozen_scenarios(json.loads((root/'configs/robustness_v2.json').read_text()))
    model=root/model_path
    expected='85f2006b010672063758259d367428f8e4a73706251e8842ac0f993f0b7782d7'
    if digest_file(model)!=expected:raise ValueError('Frozen checkpoint mismatch')
    parameters={'seed':20261109,'threads':4,'transition_penalty':10,
                'min_frames':12,'short_penalty':2}
    thresholds={'minimum_mean_delta':.01,'minimum_worst_delta':-.02}
    temporal.configure(parameters['seed'],parameters['threads'])
    checkpoint=torch.load(model,map_location='cpu',weights_only=True)
    obs,targets=load_match(root,match,'validation',protocol)
    records=[]
    for scenario in scenarios:
        changed=corrupt(obs,scenario);p=infer(checkpoint,changed)
        row={'scenario':scenario['name']}
        for name,labels in [('v2',temporal.decode(p,changed.sequence_id,parameters['transition_penalty'])),('duration_layer',duration_decode(p,changed.sequence_id,parameters['transition_penalty'],parameters['min_frames'],parameters['short_penalty']))]:
            metrics=analyze(targets.label,labels,obs.sequence_id,changed.ball_observed)['official']['metrics']
            row[name]=float(metrics['spell_f1_mean'])
        records.append(row)
        print(scenario['name'],'complete',flush=True)
    deltas=[r['duration_layer']-r['v2'] for r in records]
    result={'records':records,'mean_delta':float(np.mean(deltas)),'worst_delta':min(deltas),
            'eligible_for_further_review':bool(np.mean(deltas)>=thresholds['minimum_mean_delta'] and min(deltas)>=thresholds['minimum_worst_delta']),
            'report_version':2,'provenance':provenance,'scenarios':scenarios,
            'candidate_parameters':parameters,'review_thresholds':thresholds,
            'model_sha256':expected,'interpretation':'Reused external development match. Fixed soft-duration candidate. No public-sample tuning. No promotion.'}
    verify_provenance(provenance,root,runner,model_path,match)
    if output.exists():raise FileExistsError('Experiment already recorded')
    write_json(output,result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':run()
