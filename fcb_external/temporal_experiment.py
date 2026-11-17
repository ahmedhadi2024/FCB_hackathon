"""Predeclared CPU temporal experiment, restricted to external train/validation."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from . import team_features, temporal
from .experiment import load_match,input_fingerprint
from .features import extract as pilot_features
from .robustness import corrupt
from .model import probability,team_states,decode as pilot_decode
from .baseline import predict
from .metrics import evaluate_labels
from .schema import write_json,digest_file
from .cli import source_hashes


def infer(checkpoint,obs,chunk_frames=4096):
    if checkpoint['features']!=list(team_features.NAMES):
        raise ValueError('Temporal feature contract mismatch')
    architecture=checkpoint['network']
    net=temporal.TemporalNet(len(team_features.NAMES),architecture['channels'],architecture['dilations'])
    net.load_state_dict(checkpoint['state_dict'])
    mean=checkpoint['mean'].numpy(); scale=checkpoint['scale'].numpy()
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid checkpoint feature scaling')
    x=team_features.extract(obs)
    a=temporal.probabilities(net,(x-mean)/scale,obs.sequence_id,chunk_frames)
    b=temporal.probabilities(net,(team_features.swap(x)-mean)/scale,obs.sequence_id,chunk_frames)
    return (a+b[:,[1,0,2]])*.5


def verify_reference(root,cfg,split,reference):
    root=Path(root)
    if reference.get('model_sha256')!=digest_file(root/'models/temporal_v1.pt'):
        raise ValueError('Promotion reference model checksum mismatch')
    if reference.get('config_sha256')!=digest_file(root/'configs'/cfg['scenario_config']):
        raise ValueError('Promotion reference scenario checksum mismatch')
    expected={match:{kind:input_fingerprint(root/'artifacts'/match/kind)
                     for kind in ('observations','targets')} for match in split['validation']}
    if reference.get('inputs')!=expected:
        raise ValueError('Promotion reference validation input fingerprint mismatch')


def run(root,config_name='temporal_v1.json'):
    root=Path(root)
    config_path=root/'configs'/config_name
    cfg=json.loads(config_path.read_text())
    output_name=cfg.get('output_name','temporal_v1')
    if Path(output_name).name!=output_name or not output_name:
        raise ValueError('Simple experiment output name required')
    if config_name!='temporal_v1.json' and output_name=='temporal_v1':
        raise ValueError('New experiments must preserve the v1 release')
    if (root/'models'/f'{output_name}.pt').exists() or (root/'reports'/output_name/'validation.json').exists():
        raise FileExistsError('Experiment output already exists; choose a new experiment name, preserving releases')
    split_path=config_path.parent/cfg['split_protocol']
    split=json.loads(split_path.read_text())
    scenarios=split['robustness']
    if cfg.get('scenario_config'):
        scenarios=json.loads((config_path.parent/cfg['scenario_config']).read_text())['scenarios']
    reference=None; reference_hash=None
    if cfg.get('reference_benchmark'):
        reference_path=root/cfg['reference_benchmark']
        reference=json.loads(reference_path.read_text())
        reference_hash=digest_file(reference_path)
        verify_reference(root,cfg,split,reference)
    from download_development import development_ids
    development_ids(split)
    if cfg['network']['kernel_size']!=3:
        raise ValueError('This architecture requires kernel size 3')
    temporal.configure(cfg['seed'],cfg['training']['cpu_threads'])
    rng=np.random.default_rng(cfg['seed'])
    started=time.perf_counter(); train_inputs={}; val_inputs={}
    cache=[]; scaling=[]; class_counts=np.zeros(3,dtype=np.int64)
    for match in split['train']:
        obs,targets=load_match(root,match,'train',split)
        folder=root/'artifacts'/match
        train_inputs[match]={kind:input_fingerprint(folder/kind) for kind in ('observations','targets')}
        y=np.full(len(targets.label),-100,np.int64)
        for i,label in enumerate(('H','A','D')):
            y[targets.label==label]=i
            class_counts[i]+=np.sum(y==i)
        for condition in scenarios:
            changed=obs if condition['name']=='clean' else corrupt(obs,condition)
            x=team_features.extract(changed)
            cache.append((x,y,obs.sequence_id))
            indices=rng.choice(len(x),min(len(x),cfg['training']['scaling_sample_frames_per_condition']),replace=False)
            sample=x[indices]; scaling.extend([sample,team_features.swap(sample)])
        print('Temporal training inputs ready:',match,flush=True)
    if np.any(class_counts==0):
        raise ValueError('Training requires all three external target classes')
    scale_rows=np.concatenate(scaling)
    mean=scale_rows.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale=scale_rows.std(axis=0,dtype=np.float64).astype(np.float32); scale[scale<1e-6]=1
    del scale_rows,scaling
    arch=cfg['network']; training=cfg['training']
    net=temporal.TemporalNet(len(team_features.NAMES),arch['channels'],arch['dilations'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    weights=class_counts.sum()/(3*class_counts.astype(float))
    # Team-exchange augmentation makes the two owner class weights identical.
    weights[:2]=class_counts.sum()/(3*class_counts[:2].mean())
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),ignore_index=-100)
    history=[]
    for epoch in range(training['epochs']):
        net.train(); losses=[]
        for _ in range(training['steps_per_epoch']):
            x,y,seq=cache[int(rng.integers(len(cache)))]
            windows,labels,mask=temporal.sample_windows(x,y,seq,rng,training['batch_size'],
                training['core_frames'],net.radius,return_mask=True)
            exchange=rng.random(len(windows))<training['team_exchange_probability']
            for i in np.flatnonzero(exchange):
                windows[i]=team_features.swap(windows[i].T).T
                valid=(labels[i]==0)|(labels[i]==1)
                labels[i,valid]=1-labels[i,valid]
            if not np.any(labels!=-100):
                continue
            windows=(windows-mean[None,:,None])/scale[None,:,None]
            optimizer.zero_grad(set_to_none=True)
            logits=net(torch.from_numpy(windows),torch.from_numpy(mask))
            loss=criterion(logits,torch.from_numpy(labels))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite external training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),training['gradient_clip'])
            optimizer.step(); losses.append(float(loss.detach()))
        if not losses:
            raise ValueError('No valid external training windows')
        history.append(float(np.mean(losses)))
        print(f'Epoch {epoch+1}/{training["epochs"]}: training loss {history[-1]:.5f}',flush=True)
    del cache
    checkpoint={'format_version':1,'kind':'pytorch_temporal_conv_v1',
        'network':arch,'features':list(team_features.NAMES),
        'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),
        'state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
        'training_inputs':train_inputs,'config_sha256':digest_file(config_path),
        'split_sha256':digest_file(split_path),'source_sha256':source_hashes(),
        'torch_version':str(torch.__version__),'seed':cfg['seed']}
    pilot_path=Path(cfg.get('pilot_model',root/'models/live_dead_v1.json'))
    pilot=json.loads(pilot_path.read_text())
    penalties=cfg['selection']['transition_penalties']; conditions=[]
    for match in split['validation']:
        obs,targets=load_match(root,match,'validation',split)
        val_inputs[match]={kind:input_fingerprint(root/'artifacts'/match/kind) for kind in ('observations','targets')}
        for condition in scenarios:
            changed=obs if condition['name']=='clean' else corrupt(obs,condition)
            p=infer(checkpoint,changed,cfg['inference_chunk_frames'])
            owners=team_states(predict(changed,50),changed.sequence_id)
            dec=pilot['selected_decoding']
            labels=pilot_decode(owners,probability(pilot,pilot_features(changed)),changed.sequence_id,
                                dec['probability_threshold'],dec['half_width_frames'])
            item={'match':match,'scenario':condition['name'],
                  'pilot':evaluate_labels(targets.label,labels,obs.sequence_id),'candidates':{}}
            for penalty in penalties:
                labels=temporal.decode(p,obs.sequence_id,penalty)
                item['candidates'][str(penalty)]=evaluate_labels(targets.label,labels,obs.sequence_id)
            conditions.append(item)
            print('Temporal validation complete:',match,condition['name'],flush=True)
    def metric(item,key):
        return item['candidates'][str(key)]['metrics']['spell_f1_mean']
    primary=cfg['selection'].get('primary_scenarios',[s['name'] for s in scenarios])
    selection_conditions=[item for item in conditions if item['scenario'] in primary]
    means={str(p):float(np.mean([metric(item,p) for item in selection_conditions])) for p in penalties}
    selected=max(penalties,key=lambda p:(means[str(p)],float(np.mean([
        metric(item,p) for item in conditions if item['scenario']=='clean'])),-p))
    baseline_mean=float(np.mean([item['pilot']['metrics']['spell_f1_mean'] for item in selection_conditions]))
    deltas=[metric(item,selected)-item['pilot']['metrics']['spell_f1_mean'] for item in conditions]
    promote=means[str(selected)]>baseline_mean and min(deltas)>=-.02
    promotion_reference={'kind':'logistic_pilot','mean':baseline_mean}
    if cfg.get('reference_benchmark'):
        if digest_file(reference_path)!=reference_hash:
            raise ValueError('Promotion reference changed during training')
        verify_reference(root,cfg,split,reference)
        fixed={(item['match'],item['scenario']):item['neural']['official']['metrics']['spell_f1_mean']
               for item in reference['conditions']}
        reference_mean=float(np.mean([fixed[(item['match'],item['scenario'])] for item in selection_conditions]))
        promote=means[str(selected)]>reference_mean and all(
            metric(item,selected)>=fixed[(item['match'],item['scenario'])]-.02 for item in conditions)
        promotion_reference={'kind':'frozen_neural_v1','mean':reference_mean,'sha256':reference_hash}
    checkpoint.update(selected_transition_penalty=selected,promotion_eligible=promote,
                      status='External validation-selected challenger; sealed tests untouched')
    model_path=root/'models'/f'{output_name}.pt'; model_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint,model_path)
    report={'config':cfg,'config_sha256':digest_file(config_path),'split_sha256':digest_file(split_path),
        'training_inputs':train_inputs,'validation_inputs':val_inputs,'source_sha256':source_hashes(),
        'training_loss':history,'class_counts':class_counts.tolist(),
        'parameter_count':sum(p.numel() for p in net.parameters()),
        'candidate_mean_spellF1':means,'pilot_mean_spellF1':baseline_mean,
        'selected_transition_penalty':selected,'promotion_eligible':promote,'promotion_reference':promotion_reference,
        'conditions':conditions,'test_matches_evaluated':[],
        'model_sha256':digest_file(model_path),'elapsed_seconds':time.perf_counter()-started,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},
        'limitations':'One reused external validation match; synthetic corruption; DFL/Metrica domain shift unmeasured.'}
    write_json(root/'reports'/output_name/'validation.json',report)
    rows=['# Temporal neural pilot: external validation','',
          f'Parameters: {report["parameter_count"]}; selected transition penalty: {selected}.',
          '', '| Condition | Frozen pilot | Raw neural | Selected neural |', '|---|---:|---:|---:|']
    for item in conditions:
        rows.append(f'| {item["scenario"]} | {item["pilot"]["metrics"]["spell_f1_mean"]:.4f} | '
                    f'{metric(item,0):.4f} | {metric(item,selected):.4f} |')
    rows.extend(['',f'Mean pilot: {baseline_mean:.4f}; mean neural: {means[str(selected)]:.4f}.',
                 f'Eligible for promotion under the predeclared gate: {promote}.',
                 'Validation was used for selection, not an independent final test. No competition feedback used.'])
    (root/'reports'/output_name/'summary.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('\n'.join(rows),flush=True)
    return report
