"""Bounded external validation of decoder costs; the trained network stays frozen."""
import json
from pathlib import Path
import numpy as np
import torch
from . import temporal
from .state_decoder import decode
from .temporal_experiment import infer
from .experiment import load_match,input_fingerprint
from .robustness import corrupt
from .error_analysis import analyze
from .schema import digest_file,write_json
from .cli import source_hashes


def select_candidate(records,cfg):
    baseline=f"{cfg['baseline']['owner']}/{cfg['baseline']['dead_live']}"
    if not records or len({r['key'] for r in records})!=len(records):
        raise ValueError('Unique nonempty decoder candidate records required')
    by_key={r['key']:r for r in records}
    if baseline not in by_key: raise ValueError('Frozen baseline candidate is missing')
    expected={c['scenario'] for c in by_key[baseline]['conditions']}
    primary=set(cfg['primary_scenarios'])
    if not expected or not primary or not primary<=expected:
        raise ValueError('Complete primary validation conditions required')
    summaries={}
    for r in records:
        conditions={c['scenario']:c for c in r['conditions']}
        if len(conditions)!=len(r['conditions']) or set(conditions)!=expected:
            raise ValueError('All candidates must cover identical unique conditions')
        values=[c[k] for c in conditions.values() for k in ('spell_f1_mean','short_recall')]
        if any(v is None for v in values) or not np.isfinite(values).all():
            raise ValueError('Finite measured scores and short-possession recall required')
        summaries[r['key']]={'primary_mean':float(np.mean([c['spell_f1_mean'] for s,c in conditions.items() if s in primary])),
            'short_recall_mean':float(np.mean([c['short_recall'] for s,c in conditions.items() if s in primary])),
            'conditions':conditions}
    fixed=summaries[baseline]; gate=cfg['promotion']; eligible=[]
    for r in records:
        key=r['key']; s=summaries[key]
        s['primary_gain']=s['primary_mean']-fixed['primary_mean']
        s['short_recall_gain']=s['short_recall_mean']-fixed['short_recall_mean']
        s['worst_condition_delta']=min(c['spell_f1_mean']-fixed['conditions'][name]['spell_f1_mean']
                                       for name,c in s['conditions'].items())
        s['eligible']=key!=baseline and s['primary_gain']+1e-12>=gate['minimum_mean_spellF1_gain'] and \
            s['short_recall_gain']+1e-12>=gate['minimum_short_recall_gain'] and \
            s['worst_condition_delta']+1e-12>=-gate['maximum_condition_drop']
        if s['eligible']: eligible.append(r)
    def rank(r):
        s=summaries[r['key']]
        distance=abs(r['owner']-cfg['baseline']['owner'])+abs(r['dead_live']-cfg['baseline']['dead_live'])
        return (s['primary_mean'],s['worst_condition_delta'],s['short_recall_mean'],-distance,-r['owner'],-r['dead_live'])
    selected=max(eligible,key=rank)['key'] if eligible else baseline
    return {'selected':selected,'promoted':bool(eligible),'candidates':summaries}


def run(root):
    root=Path(root); config_path=root/'configs/decoder_v1.json'
    cfg=json.loads(config_path.read_text()); out=root/'reports/decoder_v1'
    if (out/'validation.json').exists(): raise FileExistsError('Decoder experiment report already exists')
    model_path=root/cfg['model']
    if digest_file(model_path)!=cfg['model_sha256']: raise ValueError('Frozen neural checkpoint mismatch')
    split_path=root/cfg['split_protocol']; split=json.loads(split_path.read_text())
    if cfg['validation_match'] not in split['validation'] or cfg['validation_match'] in split['sealed_test']+split['train']:
        raise ValueError('Decoder selection requires an exclusive external validation match')
    scenario_path=root/cfg['scenario_config']; scenarios=json.loads(scenario_path.read_text())['scenarios']
    temporal.configure(20261108,4)
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
    obs,targets=load_match(root,cfg['validation_match'],'validation',split)
    provenance={'model_sha256':digest_file(model_path),'config_sha256':digest_file(config_path),
                'split_sha256':digest_file(split_path),'scenario_sha256':digest_file(scenario_path),
                'source_sha256':source_hashes(),'inputs':{kind:input_fingerprint(root/'artifacts'/obs.match_id/kind)
                                                       for kind in ('observations','targets')}}
    records=[{'key':f'{owner}/{dead}','owner':owner,'dead_live':dead,'conditions':[]}
             for owner in cfg['owner_costs'] for dead in cfg['dead_live_costs']]
    for scenario in scenarios:
        changed=corrupt(obs,scenario)
        p=infer(checkpoint,changed)
        for r in records:
            labels=decode(p,changed.sequence_id,r['owner'],r['dead_live'])
            diagnostics=analyze(targets.label,labels,obs.sequence_id,changed.ball_observed)
            m=diagnostics['official']['metrics']
            item={'scenario':scenario['name'],'spell_f1_mean':m['spell_f1_mean'],'macro_f1':m['macro_f1'],
                  'gt_spells':m['n_gt_spells'],'pred_spells':m['n_pred_spells'],
                  'short_recall':diagnostics['duration_bins']['under_2s']['recall_iou_0.5'],
                  'duration_bins':diagnostics['duration_bins'],'potential_merges':diagnostics['merged_predictions']}
            r['conditions'].append(item)
            print(scenario['name'],r['key'],round(item['spell_f1_mean'],4),flush=True)
        write_json(out/'progress.json',{'status':'in_progress','provenance':provenance,'records':records})
    if digest_file(model_path)!=provenance['model_sha256'] or digest_file(config_path)!=provenance['config_sha256'] or \
            digest_file(scenario_path)!=provenance['scenario_sha256'] or source_hashes()!=provenance['source_sha256']:
        raise ValueError('Frozen experiment inputs or source changed during evaluation')
    result=select_candidate(records,cfg)
    report={'config':cfg,'provenance':provenance,'records':records,'decision':result,
            'regression_matches_read':[],'interpretation':'Development selection on one reused validation match. No new independent final evidence.'}
    write_json(out/'validation.json',report)
    print('Selected:',result['selected'],'promoted:',result['promoted'],flush=True)
    return report
