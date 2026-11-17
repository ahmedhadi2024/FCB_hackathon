"""Version-4 retraining on external matches only. No competition data is read here.

Stage 'cv'    : grouped cross-validation over the external matches. Each fold trains a
                v4-feature network, scores a grid of owner / dead-live decoder costs on the
                held-out matches, and scores the frozen v2 release on the same corrupted
                observations as a paired comparator. Writes reports/<name>/decision.json.
Stage 'final' : only if the CV decision promotes v4. Refits on every external match with the
                selected costs and writes models/<name>.pt plus a submission config.

Selection uses the ~40%-missing-ball scenarios, because the organisers document the ball as
absent on 41% of graded frames. That figure comes from the public documentation, not from
possession answers.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from . import team_features_v4, team_features_v6, temporal, state_decoder, possession_convention
from .experiment import input_fingerprint
from .feature_recovery_experiment import seed_for, scaling, infer, atomic_checkpoint
from .features import sequence_spans
from .metrics import evaluate_labels
from .robustness import corrupt
from .schema import Observations, Targets, digest_file, write_json
from .cli import source_hashes

KIND='pytorch_temporal_conv_v4'
MODULE=team_features_v4
MODULES={'v4':team_features_v4,'v6':team_features_v6}


def module_for(cfg):
    return MODULES[cfg.get('feature_module','v4')]


def validate_config(cfg):
    matches=cfg['matches']
    if len(matches)!=len(set(matches)) or len(matches)<2:
        raise ValueError('Unique external matches required')
    if any('BAR' in m or 'ATH' in m for m in matches):
        raise ValueError('Competition match is not permitted in external retraining')
    held=[]
    for fold in cfg['folds']:
        if set(fold['train'])&set(fold['validation']) or not fold['train'] or not fold['validation']:
            raise ValueError('Fold roles must be disjoint and nonempty')
        if set(fold['train']+fold['validation'])!=set(matches):
            raise ValueError('Every fold must partition the declared matches')
        held.extend(fold['validation'])
    if sorted(held)!=sorted(matches) or len({f['name'] for f in cfg['folds']})!=len(cfg['folds']):
        raise ValueError('Each match must be held out exactly once by uniquely named folds')
    names={s['name'] for s in cfg['scenarios']}
    for key in ('training_scenarios','evaluation_scenarios','selection_scenarios'):
        if not cfg[key] or len(cfg[key])!=len(set(cfg[key])) or not set(cfg[key])<=names:
            raise ValueError('Unknown or repeated scenario in '+key)
    if not set(cfg['selection_scenarios'])<=set(cfg['evaluation_scenarios']):
        raise ValueError('Selection scenarios must be evaluated')
    grid=cfg['decoder_grid']
    costs=list(grid['owner'])+list(grid['dead_live'])
    if not grid['owner'] or not grid['dead_live'] or any(not math.isfinite(c) or c<=0 for c in costs):
        raise ValueError('Positive finite decoder costs required')
    if not set(cfg['comparator']['unseen_matches'])<=set(matches):
        raise ValueError('Comparator matches must be declared external matches')
    if cfg['network']['kernel_size']!=3:
        raise ValueError('This architecture requires kernel size 3')
    frames=cfg.get('controlled_possession',{}).get('min_frames',0)
    if not isinstance(frames,int) or isinstance(frames,bool) or frames<0:
        raise ValueError('Nonnegative integer controlled-possession duration required')


def min_controlled(cfg):
    return cfg.get('controlled_possession',{}).get('min_frames',0)


def convention_labels(cfg,targets,obs):
    """External targets, rewritten to the published controlled-possession rule if configured."""
    label=np.asarray(targets.label); frames=min_controlled(cfg)
    if not frames:
        return label
    return possession_convention.absorb_short_spells(label,np.asarray(obs.sequence_id),frames)


def load_external(root,match):
    folder=Path(root)/'artifacts'/match
    if not (folder/'observations'/'metadata.json').exists():
        raise ValueError(f'Missing prepared external match {match}; run the prepare/download steps first')
    obs=Observations.load(folder/'observations')
    targets=Targets.load(folder/'targets',obs)
    if obs.match_id!=match:
        raise ValueError('Wrong external match artifact')
    return obs,targets


def integer_labels(label):
    y=np.full(len(label),-100,np.int64)
    for i,name in enumerate(('H','A','D')):
        y[label==name]=i
    return y


def fingerprint(root,match):
    return {kind:input_fingerprint(Path(root)/'artifacts'/match/kind) for kind in ('observations','targets')}


def changed(obs,scenario,role,fold_name):
    if scenario['name']=='clean':
        return obs
    seed=seed_for(role,fold_name,obs.match_id,scenario['name'])
    if scenario.get('slow_weight'):
        # Ball loss biased to slow-player moments; conditions on observations, never on targets.
        return team_features_v6.pace_mask(obs,scenario['missing_fraction'],scenario['slow_weight'],
                                          scenario.get('gap_min_frames',25),scenario.get('gap_max_frames',375),seed)
    return corrupt(obs,{**scenario,'seed':seed})


def training_arrays(root,cfg,matches,fold_name):
    """Extract features immediately so corrupted observation copies are not kept in memory."""
    scenarios=[s for s in cfg['scenarios'] if s['name'] in cfg['training_scenarios']]
    arrays=[]; inputs={}
    for match in matches:
        obs,targets=load_external(root,match)
        inputs[match]=fingerprint(root,match)
        y=integer_labels(convention_labels(cfg,targets,obs))
        for scenario in scenarios:
            x=module_for(cfg).extract(changed(obs,scenario,'train',fold_name))
            arrays.append((x,y,np.array(obs.sequence_id)))
            print('features',fold_name,match,scenario['name'],flush=True)
    return arrays,inputs


def fit(arrays,seed,arch,training,module=MODULE):
    """AdamW with cosine decay; fixed budget, final weights only (no validation peeking)."""
    temporal.configure(seed,training.get('cpu_threads',4))
    rng=np.random.default_rng(seed)
    samples=[]
    for x,_,_ in arrays:
        take=min(len(x),training['scaling_sample_frames_per_condition'])
        samples.append(x[rng.choice(len(x),take,replace=False)])
    mean,scale=scaling(samples,module)
    counts=np.bincount(np.concatenate([y[y>=0] for _,y,_ in arrays]),minlength=3)
    if np.any(counts==0):
        raise ValueError('All three classes required')
    weights=counts.sum()/(3*counts.astype(float))
    weights[:2]=counts.sum()/(3*counts[:2].mean())
    device=temporal.DEVICE
    net=temporal.TemporalNet(len(module.NAMES),arch['channels'],arch['dilations']).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=training['learning_rate'],
                                weight_decay=training['weight_decay'])
    total=training['epochs']*training['steps_per_epoch']
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=total,eta_min=training['learning_rate']*training.get('final_lr_fraction',0.05))
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),ignore_index=-100)
    rng=np.random.default_rng(seed+100000)
    history=[]; started=time.perf_counter()
    for epoch in range(training['epochs']):
        net.train(); losses=[]
        for _ in range(training['steps_per_epoch']):
            x,y,seq=arrays[int(rng.integers(len(arrays)))]
            windows,labels,mask=temporal.sample_windows(x,y,seq,rng,training['batch_size'],
                training['core_frames'],net.radius,return_mask=True)
            for i in np.flatnonzero(rng.random(len(windows))<training['team_exchange_probability']):
                windows[i]=module.swap(windows[i].T).T
                valid=(labels[i]==0)|(labels[i]==1)
                labels[i,valid]=1-labels[i,valid]
            if np.any(labels!=-100):
                windows=(windows-mean[None,:,None])/scale[None,:,None]
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(net(torch.from_numpy(windows).to(device),torch.from_numpy(mask).to(device)),torch.from_numpy(labels).to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite external training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),training['gradient_clip'])
                optimizer.step(); losses.append(float(loss.detach()))
            schedule.step()
        if not losses:
            raise ValueError('No valid external training windows')
        history.append(float(np.mean(losses)))
        print('epoch',epoch+1,'of',training['epochs'],'loss',round(history[-1],5),
              'elapsed_s',round(time.perf_counter()-started,1),flush=True)
    return {'format_version':1,'kind':KIND,'network':arch,'features':list(module.NAMES),
            'feature_module':next(k for k,v in MODULES.items() if v is module),
            'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale),
            'state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
            'training_loss':history,'seed':seed,
            'parameter_count':sum(p.numel() for p in net.parameters())}


def decode_many(probabilities,sequence_id,pairs):
    """Viterbi for several (owner, dead_live) cost pairs in one pass over the frames.

    Returns an int8 array (len(pairs), frames) with 0=H, 1=A, 2=D. Each row equals
    state_decoder.decode for that pair.
    """
    p=np.asarray(probabilities,dtype=float); pairs=np.asarray(pairs,dtype=float)
    if p.shape!=(len(sequence_id),3) or pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0):
        raise ValueError('Aligned probabilities and (owner, dead_live) cost pairs required')
    k=len(pairs); owner=pairs[:,0]; dead=pairs[:,1]
    transition=np.zeros((k,3,3))
    transition[:,0,1]=transition[:,1,0]=-owner
    for a,b in ((0,2),(1,2),(2,0),(2,1)):
        transition[:,a,b]=-dead
    result=np.empty((k,len(p)),np.int8); rows=np.arange(k)
    for lo,hi in sequence_spans(np.asarray(sequence_id)):
        emission=np.log(np.maximum(p[lo:hi],1e-8))
        back=np.empty((hi-lo,k,3),np.int8)
        score=np.repeat(emission[0][None,:],k,axis=0)
        for i in range(1,hi-lo):
            options=score[:,:,None]+transition
            back[i]=options.argmax(axis=1)
            score=options.max(axis=1)+emission[i]
        state=score.argmax(axis=1)
        for i in range(hi-lo-1,-1,-1):
            result[:,lo+i]=state
            if i:
                state=back[i,rows,state]
    return result


def grid_pairs(cfg):
    return [(float(o),float(d)) for o in cfg['decoder_grid']['owner'] for d in cfg['decoder_grid']['dead_live']]


def pair_key(pair):
    return f'{pair[0]:g}/{pair[1]:g}'


def summary(result):
    m=result['metrics']
    return {'spellF1':float(m['spell_f1_mean']),'macroF1':float(m['macro_f1']),
            'pred_spells':int(m['n_pred_spells']),'gt_spells':int(m['n_gt_spells'])}


def validate_fold(root,cfg,fold,checkpoint,reference):
    from .temporal_experiment import infer as infer_v2
    scenarios=[s for s in cfg['scenarios'] if s['name'] in cfg['evaluation_scenarios']]
    pairs=grid_pairs(cfg); names=np.array(['H','A','D']); cells=[]; inputs={}
    for match in fold['validation']:
        obs,targets=load_external(root,match)
        inputs[match]=fingerprint(root,match)
        truth=convention_labels(cfg,targets,obs); frames=min_controlled(cfg); seq=np.asarray(obs.sequence_id)
        for scenario in scenarios:
            work=changed(obs,scenario,'validation',fold['name'])
            decoded=decode_many(infer(checkpoint,work,module_for(cfg)),obs.sequence_id,pairs)
            candidates={}
            for pair,row in zip(pairs,decoded):
                predicted=names[row]
                if frames:
                    predicted=possession_convention.absorb_short_spells(predicted,seq,frames)
                candidates[pair_key(pair)]=summary(evaluate_labels(truth,predicted,obs.sequence_id))
            # The released v2 is scored exactly as it would be submitted, against the same targets.
            v2=temporal.decode(infer_v2(reference,work),obs.sequence_id,reference['selected_transition_penalty'])
            cells.append({'fold':fold['name'],'match':match,'scenario':scenario['name'],
                          'missing_ball_fraction':float((~work.ball_observed).mean()),
                          'v4':candidates,'v2':summary(evaluate_labels(truth,v2,obs.sequence_id))})
            print('scored',fold['name'],match,scenario['name'],flush=True)
    return cells,inputs


def decide(cells,cfg):
    """Pick decoder costs on the selection scenarios, then compare with frozen v2."""
    pairs=[pair_key(p) for p in grid_pairs(cfg)]
    select=[c for c in cells if c['scenario'] in cfg['selection_scenarios']]
    expected={(m,s) for m in cfg['matches'] for s in cfg['evaluation_scenarios']}
    if {(c['match'],c['scenario']) for c in cells}!=expected or len(cells)!=len(expected):
        raise ValueError('Complete, unique match/scenario score cells required')

    def match_mean(group,value):
        by_match={}
        for cell in group:
            by_match.setdefault(cell['match'],[]).append(value(cell))
        return float(np.mean([np.mean(v) for _,v in sorted(by_match.items())]))

    selection={k:match_mean(select,lambda c,k=k:c['v4'][k]['spellF1']) for k in pairs}
    overall={k:match_mean(cells,lambda c,k=k:c['v4'][k]['spellF1']) for k in pairs}
    chosen=max(pairs,key=lambda k:(selection[k],overall[k],-pairs.index(k)))
    unseen=[c for c in select if c['match'] in cfg['comparator']['unseen_matches']]
    per_match={}
    for match in cfg['comparator']['unseen_matches']:
        rows=[c for c in unseen if c['match']==match]
        per_match[match]={'v4':float(np.mean([c['v4'][chosen]['spellF1'] for c in rows])),
                          'v2':float(np.mean([c['v2']['spellF1'] for c in rows]))}
    gain=float(np.mean([v['v4']-v['v2'] for v in per_match.values()]))
    clean=[c for c in cells if c['scenario']=='clean' and c['match'] in cfg['comparator']['unseen_matches']]
    clean_delta=float(np.mean([c['v4'][chosen]['spellF1']-c['v2']['spellF1'] for c in clean])) if clean else None
    gates=cfg['comparator']
    promote=gain>=gates['minimum_mean_gain'] and all(v['v4']>v['v2'] for v in per_match.values()) and \
        (clean_delta is None or clean_delta>=-gates['maximum_clean_loss'])
    owner,dead=chosen.split('/')
    grid=cfg['decoder_grid']
    edge={'owner_at_grid_maximum':float(owner)==max(grid['owner']),
          'owner_at_grid_minimum':float(owner)==min(grid['owner']),
          'dead_live_at_grid_maximum':float(dead)==max(grid['dead_live']),
          'dead_live_at_grid_minimum':float(dead)==min(grid['dead_live'])}
    return {'selected_costs':[float(owner),float(dead)],'selection_mean_spellF1':selection,
            'all_scenario_mean_spellF1':overall,'unseen_match_selection_scores':per_match,
            'unseen_mean_gain_vs_v2':gain,'unseen_clean_delta_vs_v2':clean_delta,
            'selected_at_grid_edge':edge,'promote':bool(promote),
            'caveat':'v2 trained or was selected on every match except the comparator matches, so only '
                     'those are a fair paired comparison. Costs were chosen on the same CV cells, which '
                     'is mildly optimistic. External provider labels; no competition score is implied.'}


def provenance(cfg_path):
    return {'config_sha256':digest_file(cfg_path),'source_sha256':source_hashes(),
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)}}


def run_cv(root,config_name='retrain_v4.json'):
    root=Path(root); cfg_path=root/'configs'/config_name
    cfg=json.loads(cfg_path.read_text()); validate_config(cfg)
    name=cfg['output_name']; report_dir=root/'reports'/name; model_dir=root/'models'/name
    report_dir.mkdir(parents=True,exist_ok=True); model_dir.mkdir(parents=True,exist_ok=True)
    if (report_dir/'decision.json').exists():
        raise FileExistsError('Decision already exists; choose a new output_name to rerun')
    reference_path=root/cfg['comparator']['model']
    if digest_file(reference_path)!=cfg['comparator']['model_sha256']:
        raise ValueError('Comparator model checksum mismatch')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    stamp=provenance(cfg_path); cells=[]
    for fold in cfg['folds']:
        report_path=report_dir/(fold['name']+'.json'); model_path=model_dir/(fold['name']+'.pt')
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['config_sha256']!=stamp['config_sha256'] or report['model_sha256']!=digest_file(model_path):
                raise ValueError('Existing fold report does not match this config/model; preserve and investigate')
            cells.extend(report['cells']); print('resumed',fold['name'],flush=True); continue
        started=time.perf_counter()
        if model_path.exists():
            checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
            if checkpoint.get('config_sha256')!=stamp['config_sha256'] or checkpoint.get('fold')!=fold['name']:
                raise ValueError('Existing fold checkpoint does not match this config; preserve and investigate')
            train_inputs=checkpoint['training_inputs']
        else:
            arrays,train_inputs=training_arrays(root,cfg,fold['train'],fold['name'])
            checkpoint=fit(arrays,fold['seed'],cfg['network'],cfg['training'],module_for(cfg)); del arrays
            checkpoint.update(config_sha256=stamp['config_sha256'],fold=fold['name'],training_inputs=train_inputs)
            atomic_checkpoint(model_path,checkpoint)
        fold_cells,val_inputs=validate_fold(root,cfg,fold,checkpoint,reference)
        write_json(report_path,{**stamp,'fold':fold,'cells':fold_cells,'training_inputs':train_inputs,
                                'validation_inputs':val_inputs,'model_sha256':digest_file(model_path),
                                'training_loss':checkpoint['training_loss'],
                                'parameter_count':checkpoint['parameter_count'],
                                'elapsed_seconds':time.perf_counter()-started})
        cells.extend(fold_cells)
    decision=decide(cells,cfg); decision.update(stamp)
    write_json(report_dir/'decision.json',decision)
    print(json.dumps({k:decision[k] for k in ('selected_costs','unseen_match_selection_scores',
          'unseen_mean_gain_vs_v2','unseen_clean_delta_vs_v2','selected_at_grid_edge','promote')},indent=2),flush=True)
    return decision


def run_final(root,config_name='retrain_v4.json'):
    root=Path(root); cfg_path=root/'configs'/config_name
    cfg=json.loads(cfg_path.read_text()); validate_config(cfg)
    name=cfg['output_name']; stamp=provenance(cfg_path)
    decision=json.loads((root/'reports'/name/'decision.json').read_text())
    if decision['config_sha256']!=stamp['config_sha256']:
        raise ValueError('Decision was produced by a different config')
    if not decision['promote']:
        raise ValueError('CV decision does not promote v4; keep the v2 release')
    model_path=root/'models'/(name+'.pt'); submission_path=root/'configs'/f'submission_{name}.json'
    if model_path.exists() or submission_path.exists():
        raise FileExistsError('Final v4 model or submission config already exists')
    arrays,train_inputs=training_arrays(root,cfg,cfg['matches'],'final')
    checkpoint=fit(arrays,cfg['final_refit_seed'],cfg['network'],cfg['training'],module_for(cfg))
    checkpoint.update(config_sha256=stamp['config_sha256'],training_inputs=train_inputs,
                      selected_transition_costs=decision['selected_costs'],
                      min_controlled_frames=min_controlled(cfg),
                      status='External-trained v4 refit on all external matches; no held-out score exists for this exact checkpoint')
    atomic_checkpoint(model_path,checkpoint)
    submission=json.loads((root/'configs'/cfg['submission_template']).read_text())
    submission.pop('smart_tag_candidate',None)
    submission.update(model=f'../models/{name}.pt',model_kind=KIND,model_sha256=digest_file(model_path))
    if cfg.get('kickoff_dead_rule'):
        submission['kickoff_dead_rule']=True
    if cfg.get('submission_min_controlled_frames'):
        submission['min_controlled_frames']=cfg['submission_min_controlled_frames']
    write_json(submission_path,submission)
    print('Final v4 model:',model_path,'\nSubmission config:',submission_path,flush=True)
    return checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('cv','final'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--config',default='retrain_v4.json')
    args=parser.parse_args()
    (run_cv if args.stage=='cv' else run_final)(args.root,args.config)
