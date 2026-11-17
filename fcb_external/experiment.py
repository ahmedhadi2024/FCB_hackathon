"""Frozen external live/dead pilot. Sealed matches cannot enter this runner."""
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
from .schema import Observations,Targets,write_json,digest_file
from .cli import source_hashes,directory_hashes
from .baseline import predict
from .features import extract,corrupt,NAMES
from .model import fit,probability,team_states,decode
from .metrics import evaluate_labels


def input_fingerprint(directory):
    directory=Path(directory)
    result=directory_hashes(directory)
    result['metadata.json']=digest_file(directory/'metadata.json')
    prepared=directory.parent/'manifest.json'
    if prepared.exists():
        result['../manifest.json']=digest_file(prepared)
    return result


def load_match(root,match,role,protocol):
    if role not in ('train','validation') or match not in protocol[role] or match in protocol['sealed_test']:
        raise ValueError('Match is sealed or not allowed for this role')
    folder=Path(root)/'artifacts'/match
    obs=Observations.load(folder/'observations')
    targets=Targets.load(folder/'targets',obs)
    if obs.match_id!=match:
        raise ValueError('Wrong external match artifact')
    return obs,targets


def run(root):
    root=Path(root)
    protocol_path=root/'configs/experiment_v1.json'
    protocol=json.loads(protocol_path.read_text())
    from download_development import development_ids
    development_ids(protocol)  # Enforce disjoint roles before any file read.
    for match in protocol['train']+protocol['validation']:
        if not (root/'artifacts'/match/'manifest.json').exists():
            raise ValueError(f'Missing prepared external match {match}; run prepare_development.py first')
    rng=np.random.default_rng(protocol['model']['seed'])
    xs=[]; ys=[]; training_inputs={}
    for match in protocol['train']:
        obs,targets=load_match(root,match,'train',protocol)
        valid=np.flatnonzero(targets.valid)
        take=min(len(valid),protocol['model']['max_sampled_frames_per_match'])
        indices=np.sort(rng.choice(valid,size=take,replace=False))
        for condition in protocol['robustness']:
            if condition['name'] not in protocol['training_scenarios']:
                continue
            changed=obs if condition['name']=='clean' else corrupt(obs,condition)
            xs.append(extract(changed)[indices]); ys.append((targets.label[indices]=='D').astype(float))
        training_inputs[match]={'observations':input_fingerprint(root/'artifacts'/match/'observations'),
                               'targets':input_fingerprint(root/'artifacts'/match/'targets')}
        print('Training features ready:',match,flush=True)
    model=fit(np.concatenate(xs),np.concatenate(ys),l2=protocol['model']['l2'],
              iterations=protocol['model']['max_iterations'])
    model['features']=list(NAMES)
    print('Live/dead model fitted on external training matches only.',flush=True)
    candidates=list(itertools.product(protocol['selection']['probability_thresholds'],
                        protocol['selection']['probability_smoothing_half_widths']))
    scores={f'{threshold}/{width}':[] for threshold,width in candidates}
    clean_scores={key:[] for key in scores}
    controls=[]; provenance={}
    for match in protocol['validation']:
        obs,targets=load_match(root,match,'validation',protocol)
        provenance[match]={'observations':input_fingerprint(root/'artifacts'/match/'observations'),
                           'targets':input_fingerprint(root/'artifacts'/match/'targets')}
        for condition in protocol['robustness']:
            changed=obs if condition['name']=='clean' else corrupt(obs,condition)
            x=extract(changed); dead=probability(model,x)
            reference=predict(changed,protocol['ownership_baseline_half_width'])
            owners=team_states(reference,changed.sequence_id)
            reference_metrics=evaluate_labels(targets.label,reference,changed.sequence_id)
            propagated_metrics=evaluate_labels(targets.label,owners,changed.sequence_id)
            entry={'match':match,'scenario':condition['name'],'reference':reference_metrics,
                   'owner_propagation_control':propagated_metrics,'candidates':{}}
            for threshold,width in candidates:
                key=f'{threshold}/{width}'
                pred=decode(owners,dead,changed.sequence_id,threshold,width)
                result=evaluate_labels(targets.label,pred,changed.sequence_id)
                entry['candidates'][key]=result
                scores[key].append(result['metrics']['spell_f1_mean'])
                if condition['name']=='clean':
                    clean_scores[key].append(result['metrics']['spell_f1_mean'])
            controls.append(entry)
            print('Validation condition complete:',match,condition['name'],flush=True)
    def rank(candidate):
        threshold,width=candidate; key=f'{threshold}/{width}'
        return (float(np.mean(scores[key])),float(np.mean(clean_scores[key])),
                -width,-abs(threshold-0.5))
    threshold,width=max(candidates,key=rank)
    model['selected_decoding']={'probability_threshold':threshold,'half_width_frames':width}
    model['protocol_sha256']=digest_file(protocol_path)
    model['training_inputs']=training_inputs
    model['source_sha256']=source_hashes()
    model['status']='PILOT selected using external validation; sealed tests NOT evaluated'
    output=root/'reports/live_dead_v1'
    write_json(root/'artifacts/live_dead_v1/model.json',model)
    report={'protocol':protocol,'protocol_sha256':digest_file(protocol_path),
            'selected':model['selected_decoding'],'training_inputs':training_inputs,
            'validation_inputs':provenance,'source_sha256':source_hashes(),
            'candidate_mean_spellF1':{key:float(np.mean(value)) for key,value in scores.items()},
            'conditions':controls,'test_matches_evaluated':[],
            'limitations':'One validation match, reused for selection; corruption conditions are synthetic. '
                          'DFL/Metrica label and observation shifts remain unmeasured.'}
    try:
        report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        report['git_commit']=None
    write_json(output/'validation.json',report)
    key=f'{threshold}/{width}'
    rows=['# First external live/dead pilot','',
          f'Selected probability threshold: {threshold}; centred probability half-width: {width} frames.',
          '', '| Condition | Reference spellF1 | Propagation control | Model spellF1 | Model D F1 |',
          '|---|---:|---:|---:|---:|']
    for entry in controls:
        chosen=entry['candidates'][key]
        rows.append(f'| {entry["scenario"]} | {entry["reference"]["metrics"]["spell_f1_mean"]:.4f} | '
                    f'{entry["owner_propagation_control"]["metrics"]["spell_f1_mean"]:.4f} | '
                    f'{chosen["metrics"]["spell_f1_mean"]:.4f} | {chosen["per_class"]["D"]["f1"]:.4f} |')
    rows.extend(['','Validation results were used for selection; they are not an independent final test.',
                 'Sealed test matches and competition data were not opened.'])
    (output/'summary.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('\n'.join(rows),flush=True)
    return report
