"""Diagnose the frozen external validation candidate; never reads competition data."""
from pathlib import Path
import json
import torch
from fcb_external import temporal
from fcb_external.temporal_experiment import infer
from fcb_external.experiment import load_match
from fcb_external.features import corrupt
from fcb_external.error_analysis import analyze
from fcb_external.schema import write_json,digest_file


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    protocol=json.loads((root/'configs/experiment_v1.json').read_text())
    temporal.configure(20261108,4)
    model=torch.load(root/'models/temporal_v1.pt',map_location='cpu',weights_only=True)
    results=[]
    for match in protocol['validation']:
        obs,targets=load_match(root,match,'validation',protocol)
        for scenario in protocol['robustness']:
            changed=obs if scenario['name']=='clean' else corrupt(obs,scenario)
            labels=temporal.decode(infer(model,changed),obs.sequence_id,model['selected_transition_penalty'])
            item={'match':match,'scenario':scenario['name'],**analyze(targets.label,labels,obs.sequence_id,changed.ball_observed)}
            results.append(item)
            print(match,scenario['name'],item['duration_bins'],flush=True)
    write_json(root/'reports/robustness_v2/error_analysis.json',
               {'model_sha256':digest_file(root/'models/temporal_v1.pt'),'conditions':results})
