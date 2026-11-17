"""Explicit final-test release; development sealed-role guards remain unchanged."""
from pathlib import Path
import json
import numpy as np
import torch
from .schema import Observations,Targets,digest_file,write_json
from .cli import source_hashes
from .experiment import input_fingerprint
from .robustness import corrupt
from .temporal_experiment import infer
from . import temporal
from .baseline import predict
from .metrics import evaluate_labels


def deployment_files(root):
    root=Path(root)
    names={'.gitattributes','run_submission.py','run_final_evaluation.py','run_robustness.py','download_final_evaluation.py',
           'download_idsse.py','requirements-cpu-lock.txt','configs/experiment_v1.json',
           'configs/robustness_v2.json','configs/submission_temporal.json','configs/submission_selected.json',
           'models/live_dead_v1.json',
           'fcb_external/__init__.py','fcb_external/final_evaluation.py',
           'vendor/official/evaluation/possession_metrics.py'}
    names.update(p.relative_to(root).as_posix() for p in (root/'fcb_external').glob('*.py'))
    names.update(p.relative_to(root).as_posix() for p in root.glob('*.py'))
    names.update(p.relative_to(root).as_posix() for p in (root/'configs').glob('*.json')
                 if not p.name.startswith('idsse_') and p.name!='final_evaluation_v1.json')
    names.update(p.relative_to(root).as_posix() for p in (root/'vendor/official').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    return sorted(names)


def verify_frozen(root,cfg):
    root=Path(root)
    if cfg.get('selection_closed') is not True:
        raise ValueError('Development selection must be closed before final evaluation')
    if digest_file(root/cfg['selected_model'])!=cfg['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    if set(cfg.get('frozen_files',{}))!=set(deployment_files(root)):
        raise ValueError('Incomplete or stale deployment freeze manifest')
    for name,expected in cfg['frozen_files'].items():
        if digest_file(root/name)!=expected:
            raise ValueError('Frozen file checksum mismatch: '+name)
    from importlib.metadata import version
    for line in (root/'requirements-cpu-lock.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name,expected=line.split('==',1)
            if version(name)!=expected:
                raise ValueError('Frozen runtime dependency mismatch: '+name)


def load_final(root,match,cfg,protocol):
    if match not in cfg['matches'] or match not in protocol['sealed_test'] or \
            match in protocol['train'] or match in protocol['validation']:
        raise ValueError('Match is not an explicitly released final-test match')
    folder=Path(root)/'artifacts'/match
    obs=Observations.load(folder/'observations'); targets=Targets.load(folder/'targets',obs)
    if obs.match_id!=match:
        raise ValueError('Wrong final-test match identity')
    return obs,targets


def run(root,config_path):
    root=Path(root); cfg=json.loads(Path(config_path).read_text())
    verify_frozen(root,cfg)
    protocol=json.loads((root/'configs/experiment_v1.json').read_text())
    scenarios=json.loads((root/'configs/robustness_v2.json').read_text())['scenarios']
    selected={s['name']:s for s in scenarios}
    temporal.configure(20261108,4)
    checkpoint=torch.load(root/cfg['selected_model'],map_location='cpu',weights_only=True)
    pilot=json.loads((root/'models/live_dead_v1.json').read_text())
    from run_robustness import pilot_labels
    items=[]; inputs={}
    for match in cfg['matches']:
        obs,targets=load_final(root,match,cfg,protocol)
        inputs[match]={k:input_fingerprint(root/'artifacts'/match/k) for k in ('observations','targets')}
        for name in cfg['scenarios']:
            changed=corrupt(obs,selected[name])
            labels=temporal.decode(infer(checkpoint,changed),changed.sequence_id,checkpoint['selected_transition_penalty'])
            pilot_pred,_=pilot_labels(pilot,changed)
            row={'match':match,'scenario':name,'missing_ball_fraction':float((~changed.ball_observed).mean()),
                 'selected':evaluate_labels(targets.label,labels,obs.sequence_id),
                 'pilot':evaluate_labels(targets.label,pilot_pred,obs.sequence_id),
                 'reference_smooth50':evaluate_labels(targets.label,predict(changed,50),obs.sequence_id)}
            items.append(row)
            print(match,name, 'selected',round(row['selected']['metrics']['spell_f1_mean'],4),
                  'pilot',round(row['pilot']['metrics']['spell_f1_mean'],4),flush=True)
    report={'protocol':cfg,'protocol_sha256':digest_file(config_path),'inputs':inputs,'conditions':items,
            'mean_match_spellF1':{name:float(np.mean([r['selected']['metrics']['spell_f1_mean'] for r in items if r['scenario']==name])) for name in cfg['scenarios']},
            'interpretation':'Independent external final evaluation after selection freeze. Means are descriptive unweighted match means, not pooled official scores. Tests are now consumed. No competition-score claim.'}
    write_json(root/'reports/final_evaluation_v1/results.json',report)
    return report
