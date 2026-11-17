"""Fixed external stress benchmark and matched decoder control. No competition inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from fcb_external import temporal
from fcb_external.temporal_experiment import infer
from fcb_external.experiment import load_match,input_fingerprint
from fcb_external.robustness import corrupt
from fcb_external.features import extract
from fcb_external.baseline import predict
from fcb_external.model import probability,team_states,decode
from fcb_external.error_analysis import analyze
from fcb_external.metrics import evaluate_labels
from fcb_external.schema import write_json,digest_file
from fcb_external.cli import source_hashes


def pilot_labels(model,obs):
    owners=team_states(predict(obs,50),obs.sequence_id)
    dead=probability(model,extract(obs))
    selected=model['selected_decoding']
    labels=decode(owners,dead,obs.sequence_id,selected['probability_threshold'],selected['half_width_frames'])
    h=np.where(owners=='H',1.,np.where(owners=='A',0.,.5))
    emissions=np.column_stack([(1-dead)*h,(1-dead)*(1-h),dead])
    control=temporal.decode(emissions,obs.sequence_id,5)
    return labels,control


def run(root,model_path,out):
    root=Path(root); model_path=Path(model_path); out=Path(out)
    protocol=json.loads((root/'configs/experiment_v1.json').read_text())
    cfg_path=root/'configs/robustness_v2.json'; cfg=json.loads(cfg_path.read_text())
    temporal.configure(20261108,4)
    model=torch.load(model_path,map_location='cpu',weights_only=True)
    pilot=json.loads((root/'models/live_dead_v1.json').read_text())
    results=[]; inputs={}
    for match in protocol['validation']:
        obs,targets=load_match(root,match,'validation',protocol)
        inputs[match]={kind:input_fingerprint(root/'artifacts'/match/kind) for kind in ('observations','targets')}
        for scenario in cfg['scenarios']:
            changed=corrupt(obs,scenario)
            labels=temporal.decode(infer(model,changed),obs.sequence_id,model['selected_transition_penalty'])
            a,b=pilot_labels(pilot,changed)
            item={'match':match,'scenario':scenario['name'],'missing_ball_fraction':float((~changed.ball_observed).mean()),
                  'neural':analyze(targets.label,labels,obs.sequence_id,changed.ball_observed),
                  'pilot':evaluate_labels(targets.label,a,obs.sequence_id),
                  'pilot_viterbi_control':evaluate_labels(targets.label,b,obs.sequence_id)}
            results.append(item)
            print(match,scenario['name'],'missing',round(item['missing_ball_fraction'],3),
                  'neural',round(item['neural']['official']['metrics']['spell_f1_mean'],4),
                  'pilot',round(item['pilot']['metrics']['spell_f1_mean'],4),
                  'control',round(item['pilot_viterbi_control']['metrics']['spell_f1_mean'],4),flush=True)
    report={'config':cfg,'config_sha256':digest_file(cfg_path),'model_sha256':digest_file(model_path),
            'inputs':inputs,'source_sha256':source_hashes(),'conditions':results,'test_matches_evaluated':[]}
    write_json(out,report)
    return report


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=root/'models/temporal_v1.pt')
    p.add_argument('--out',type=Path,default=root/'reports/robustness_v2/frozen.json')
    args=p.parse_args(); run(root,args.model,args.out)
