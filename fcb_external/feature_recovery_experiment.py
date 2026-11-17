"""Paired grouped external experiment. No competition or old final data access."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import platform
import os
import tempfile
import numpy as np
import torch
from . import team_features, team_features_v3, temporal
from .experiment import load_match, input_fingerprint
from .robustness import corrupt
from .features import sequence_spans
from .schema import write_json, digest_file
from .metrics import evaluate_labels
from .cli import source_hashes


def seed_for(role,fold,match,scenario):
    return int.from_bytes(hashlib.sha256('|'.join([role,fold,match,scenario]).encode()).digest()[:4],'little')


def validate_fold(fold):
    if set(fold['train'])&set(fold['validation']):raise ValueError('Overlapping fold roles')
    if set(fold['train']+fold['validation'])&{'J03WN1','J03WR9'}:raise ValueError('Old final matches excluded')


def scaling(xs,module):
    rows=np.concatenate([v for x in xs for v in (x,module.swap(x))])
    mean=rows.mean(0,dtype=np.float64).astype(np.float32)
    scale=rows.std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1
    return mean,scale


def fit(cache,module,seed,arch,training):
    temporal.configure(seed,training.get('cpu_threads',4));rng=np.random.default_rng(seed)
    started=time.perf_counter();arrays=[]
    for index,(obs,y) in enumerate(cache):
        x=module.extract(obs);arrays.append((x,y,obs.sequence_id))
        print('features',obs.match_id,index+1,'of',len(cache),'bytes',x.nbytes,'elapsed_s',round(time.perf_counter()-started,2),flush=True)
    samples=[]
    for x,_,_ in arrays:
        indices=rng.choice(len(x),min(len(x),training.get('scaling_sample_frames_per_condition',10000)),replace=False)
        samples.append(x[indices])
    mean,scale=scaling(samples,module)
    # Reset sampling stream independently of network width and scaler sampling.
    rng=np.random.default_rng(seed+100000)
    counts=np.bincount(np.concatenate([y[y>=0] for _,y,_ in arrays]),minlength=3)
    if np.any(counts==0):raise ValueError('All three classes required')
    weights=counts.sum()/(3*counts.astype(float));weights[:2]=counts.sum()/(3*counts[:2].mean())
    net=temporal.TemporalNet(len(module.NAMES),arch['channels'],arch['dilations'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),ignore_index=-100)
    history=[]
    for epoch in range(training['epochs']):
        net.train();losses=[]
        for _ in range(training['steps_per_epoch']):
            x,y,seq=arrays[int(rng.integers(len(arrays)))]
            windows,labels,mask=temporal.sample_windows(x,y,seq,rng,training['batch_size'],training['core_frames'],net.radius,return_mask=True)
            exchange=rng.random(len(windows))<training['team_exchange_probability']
            for i in np.flatnonzero(exchange):
                windows[i]=module.swap(windows[i].T).T
                valid=(labels[i]==0)|(labels[i]==1);labels[i,valid]=1-labels[i,valid]
            if not np.any(labels!=-100):continue
            windows=(windows-mean[None,:,None])/scale[None,:,None]
            optimizer.zero_grad(set_to_none=True)
            logits=net(torch.from_numpy(windows),torch.from_numpy(mask))
            loss=criterion(logits,torch.from_numpy(labels))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),training['gradient_clip']);optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:raise ValueError('No training windows')
        history.append(float(np.mean(losses)));print('epoch',epoch+1,'loss',history[-1],'elapsed_s',round(time.perf_counter()-started,2),flush=True)
    return dict(network=arch,features=list(module.NAMES),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},training_loss=history,seed=seed)


def infer(checkpoint,obs,module,chunk_frames=4096):
    if checkpoint['features']!=list(module.NAMES):raise ValueError('Feature contract mismatch')
    mean=checkpoint['mean'].numpy();scale=checkpoint['scale'].numpy()
    if mean.shape!=(len(module.NAMES),) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid scaling')
    arch=checkpoint['network'];net=temporal.TemporalNet(len(module.NAMES),arch['channels'],arch['dilations']);net.load_state_dict(checkpoint['state_dict'])
    x=module.extract(obs)
    a=temporal.probabilities(net,(x-mean)/scale,obs.sequence_id,chunk_frames)
    b=temporal.probabilities(net,(module.swap(x)-mean)/scale,obs.sequence_id,chunk_frames)
    return (a+b[:,[1,0,2]])*.5


AUDITS=('false_stationary_ball','ball_offset_3m','home_extra_tracks','home_visibility_loss','no_ball_first_period')


def audit(obs,name):
    xy=obs.ball_xy_m.copy();pxy=obs.player_xy_m.copy();teams=obs.player_team.copy();burst=np.zeros(len(xy),bool)
    for lo,hi in sequence_spans(obs.sequence_id):
        for start in range(lo+250,hi,1500):burst[start:min(start+250,hi)]=True
    if name=='false_stationary_ball':xy[burst]=[0,0]
    elif name=='ball_offset_3m':xy+=np.array([3,0],np.float32)
    elif name=='home_extra_tracks':
        slots=np.flatnonzero(teams==0);selected=slots[np.argsort(obs.player_observed[:,slots].sum(0),kind='stable')[-3:]]
        pxy=np.concatenate([pxy,pxy[:,selected]],1);teams=np.r_[teams,teams[selected]]
    elif name=='home_visibility_loss':pxy[np.ix_(burst,teams==0)]=np.nan
    elif name=='no_ball_first_period':xy[obs.period==obs.period[0]]=np.nan
    else:raise ValueError('Unknown audit')
    return replace(obs,ball_xy_m=xy,ball_observed=np.isfinite(xy).all(-1),player_xy_m=pxy,player_observed=np.isfinite(pxy).all(-1),player_team=teams).validate()


def validate_protocol(cfg):
    matches=cfg['matches'];names=cfg['evaluation_order']
    if len(matches)!=5 or len(set(matches))!=5 or len(names)!=18 or len(set(names))!=18:
        raise ValueError('Unique five-match, 18-condition protocol required')
    generated=[s['name'] for s in cfg['scenarios']]+cfg['audit_scenarios']+[s['name'] for s in cfg['extra_evaluation_seeds']]
    if len(generated)!=len(set(generated)) or set(generated)!=set(names) or tuple(cfg['audit_scenarios'])!=AUDITS:
        raise ValueError('Scenario definitions disagree')
    if len(cfg['folds'])!=3 or len({f['name'] for f in cfg['folds']})!=3:raise ValueError('Three unique folds required')
    held=[]
    for fold in cfg['folds']:
        validate_fold(fold)
        if any(len(fold[r])!=len(set(fold[r])) for r in ('train','validation')) or not fold['validation'] or set(fold['train']+fold['validation'])!=set(matches):raise ValueError('Invalid fold partition')
        held.extend(fold['validation'])
    if sorted(held)!=sorted(matches):raise ValueError('Each match must be held out once')
    groups=[cfg['primary'],cfg['missing_ball_family']]+list(cfg['stress_families'].values())
    if any(not g or len(g)!=len(set(g)) or not set(g)<=set(names) for g in groups):raise ValueError('Invalid score family')
    if set().union(*map(set,cfg['stress_families'].values()))!=set(names):raise ValueError('Every condition needs a family gate')
    if any(not np.isfinite(v) or v<0 for v in cfg['gates'].values()):raise ValueError('Invalid gate threshold')


def assess_gates(scores,cfg):
    validate_protocol(cfg)
    matches=sorted(cfg['matches']);names=sorted(cfg['evaluation_order'])
    owner={m:f['name'] for f in cfg['folds'] for m in f['validation']}
    expected={(v,m,n) for v in ('baseline','geometry') for m in matches for n in names}
    lookup={}
    for cell in scores.get('cells',[]):
        key=(cell['variant'],cell['match'],cell['scenario'])
        value=float(cell['spellF1'])
        if key not in expected or key in lookup or cell['fold']!=owner.get(cell['match']) or not np.isfinite(value):raise ValueError('Invalid, duplicate or misassigned score cell')
        lookup[key]=value
    if set(lookup)!=expected:raise ValueError('Complete named score cells required')
    delta={m:{n:lookup['geometry',m,n]-lookup['baseline',m,n] for n in names} for m in matches}
    average=lambda m,group:float(np.mean([delta[m][n] for n in sorted(group)]))
    primary={m:average(m,cfg['primary']) for m in matches}
    missing={m:average(m,cfg['missing_ball_family']) for m in matches}
    folds={f['name']:float(np.mean([primary[m] for m in sorted(f['validation'])])) for f in sorted(cfg['folds'],key=lambda f:f['name'])}
    families={f:float(np.mean([average(m,g) for m in matches])) for f,g in sorted(cfg['stress_families'].items())}
    t=cfg['gates'];values=np.array(list(primary.values()));miss=np.array(list(missing.values()))
    gates=dict(mean_gain=bool(values.mean()>=t['minimum_mean_gain']),four_matches=bool((values>0).sum()>=t['minimum_improved_matches']),three_folds=bool(all(v>0 for v in folds.values())),match_floor=bool(values.min()>=-t['maximum_match_loss']),without_best=bool(np.sort(values)[:-1].mean()>0),family_floor=bool(min(families.values())>=-t['maximum_family_loss']),cell_floor=bool(min(v for row in delta.values() for v in row.values())>=-t['maximum_cell_loss']),missing_mean=bool(miss.mean()>0),missing_four=bool((miss>0).sum()>=t['minimum_missing_improved_matches']))
    return dict(passed=all(gates.values()),gates=gates,match_delta=primary,fold_delta=folds,mean_delta=float(values.mean()),missing_delta=missing,family_delta=families)


def verify_resume(report,protocol_hash,sources,training_inputs,validation_inputs,model_hash):
    expected=dict(protocol_hash=protocol_hash,sources=sources,training_inputs=training_inputs,validation_inputs=validation_inputs,model_hash=model_hash)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Resume provenance mismatch')


def atomic_checkpoint(path,checkpoint):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve existing checkpoint')
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    os.close(fd)
    temporary=Path(name)
    try:
        torch.save(checkpoint,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def load_trained(path,identity):
    try: checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as error:raise ValueError('Unreadable checkpoint; preserve and investigate') from error
    if checkpoint.get('run_identity')!=identity or len(checkpoint.get('training_loss',[]))!=identity['terminal_epoch']:
        raise ValueError('Trained checkpoint provenance mismatch')
    return checkpoint


def validation_cases(obs,scenarios,extras,change):
    for spec in scenarios:
        yield spec['name'],change(obs,spec)
    for name in AUDITS:
        yield name,audit(obs,name)
    for spec in extras:
        yield spec['name'],corrupt(obs,spec)


def run(root,config_name='feature_recovery_v3.json'):
    root=Path(root);cfg_path=root/'configs'/config_name;cfg=json.loads(cfg_path.read_text());validate_protocol(cfg);protocol_hash=digest_file(cfg_path)
    sources=source_hashes();sources['official_metric']=digest_file(Path(__file__).resolve().parent.parent/'vendor/official/evaluation/possession_metrics.py');sources['runtime']={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)};report_dir=root/'reports/feature_recovery_v3';model_dir=root/'models/feature_recovery_v3'
    report_dir.mkdir(parents=True,exist_ok=True);model_dir.mkdir(parents=True,exist_ok=True)
    def get(match,fold,role):
        validate_fold(fold)
        if match not in cfg['matches']:raise ValueError('Unapproved development match')
        return load_match(root,match,role,{**fold,'sealed_test':['J03WN1','J03WR9']})
    def changed(obs,s,fold,role):
        return obs if s['name']=='clean' else corrupt(obs,{**s,'seed':seed_for(role,fold['name'],obs.match_id,s['name'])})
    all_reports=[]
    extras=[]
    for item in cfg['extra_evaluation_seeds']:
        spec=dict(next(s for s in cfg['scenarios'] if s['name']==item['base']))
        spec.update(name=item['name'],seed=item['seed']);extras.append(spec)
    for fold in cfg['folds']:
        validate_fold(fold)
        cache=[];fingerprints={}
        for match in fold['train']:
            obs,target=get(match,fold,'train');y=np.full(len(target.label),-100,np.int64)
            for i,label in enumerate(('H','A','D')):y[target.label==label]=i
            fingerprints[match]={k:input_fingerprint(root/'artifacts'/match/k) for k in ('observations','targets')}
            for s in cfg['scenarios']:
                cache.append((changed(obs,s,fold,'train'),y))
                print('prepared',fold['name'],match,s['name'],flush=True)
        for variant,module in [('baseline',team_features),('geometry',team_features_v3)]:
            name=fold['name']+'_'+variant;rp=report_dir/(name+'.json');mp=model_dir/(name+'.pt')
            if rp.exists():
                report=json.loads(rp.read_text())
                validation_fingerprints={m:{k:input_fingerprint(root/'artifacts'/m/k) for k in ('observations','targets')} for m in fold['validation']}
                verify_resume(report,protocol_hash,sources,fingerprints,validation_fingerprints,digest_file(mp))
                all_reports.append(report);continue
            identity=dict(name=name,protocol_hash=protocol_hash,sources=sources,training_inputs=fingerprints,
                          terminal_epoch=cfg['training']['epochs'])
            started=time.perf_counter()
            if mp.exists():
                checkpoint=load_trained(mp,identity)
                print('resume validation',name,flush=True)
            else:
                if list(model_dir.glob(mp.name+'.*.partial')):
                    raise ValueError('Partial checkpoint exists; preserve and investigate')
                print('fit',name,flush=True)
                checkpoint=fit(cache,module,fold['seed'],cfg['network'],cfg['training'])
                checkpoint.update(protocol_hash=protocol_hash,run_identity=identity)
                atomic_checkpoint(mp,checkpoint)
                print('checkpoint saved',name,'elapsed_s',round(time.perf_counter()-started,2),flush=True)
            cells=[];validation_inputs={}
            for match in fold['validation']:
                obs,target=get(match,fold,'validation')
                validation_inputs[match]={k:input_fingerprint(root/'artifacts'/match/k) for k in ('observations','targets')}
                for scenario,corrupted in validation_cases(obs,cfg['scenarios'],extras,lambda o,s:changed(o,s,fold,'validation')):
                    print('validation start',name,match,scenario,'elapsed_s',round(time.perf_counter()-started,2),flush=True)
                    p=infer(checkpoint,corrupted,module);labels=temporal.decode(p,obs.sequence_id,cfg['penalty'])
                    metrics=evaluate_labels(target.label,labels,obs.sequence_id)
                    cells.append(dict(match=match,scenario=scenario,official=metrics));print('score',name,match,scenario,'elapsed_s',round(time.perf_counter()-started,2),flush=True)
                    del corrupted,p,labels
            report=dict(name=name,protocol_hash=protocol_hash,sources=sources,training_inputs=fingerprints,validation_inputs=validation_inputs,model_hash=digest_file(mp),cells=cells,elapsed_seconds=time.perf_counter()-started,training_loss=checkpoint['training_loss'])
            write_json(rp,report);all_reports.append(report)
        del cache
    score_cells=[]
    for report in all_reports:
        fold,variant=report['name'].split('_',1)
        for cell in report['cells']:
            score_cells.append(dict(variant=variant,fold=fold,match=cell['match'],scenario=cell['scenario'],spellF1=cell['official']['metrics']['spell_f1_mean']))
    result=assess_gates({'cells':score_cells},cfg)
    result.update(cells=score_cells,scenarios=cfg['evaluation_order'],matches=cfg['matches'],protocol_hash=protocol_hash,evidence='paired development CV; no fresh final confirmation')
    write_json(report_dir/'decision.json',result);print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--config',default='feature_recovery_v3.json');args=parser.parse_args();run(args.root,args.config)
