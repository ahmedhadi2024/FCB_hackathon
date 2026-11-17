"""Target-free tactical inference and strictly aligned official-format export.

When run on competition assets, redirect stdout/stderr to a LOCAL ignored log.
Never supply that log, asset contents or generated CSV to a hosted AI service.
"""
import csv
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
import numpy as np
from . import atd
from .baseline import predict
from .features import extract, NAMES
from .metrics import load_official
from .model import probability, team_states, decode
from .schema import digest_file, write_json
from .cli import source_hashes


ENSEMBLE_KIND='pytorch_temporal_ensemble_v1'
# Kinds decoded with the three-state decoder, the controlled-possession filter and the kickoff rule.
STATE_KINDS=('pytorch_temporal_conv_v4',ENSEMBLE_KIND)
TEMPORAL_KINDS=('pytorch_temporal_conv_v1',)+STATE_KINDS


def timeline_blocks(n_frames, active_blocks):
    """Return complete half-open video blocks, with period 0 outside play."""
    if not isinstance(n_frames,int) or isinstance(n_frames,bool) or n_frames<1:
        raise ValueError('Positive integer frame count required')
    blocks=[]; cursor=0; periods=set()
    for period,start,end in active_blocks:
        if any(not isinstance(x,(int,np.integer)) or isinstance(x,(bool,np.bool_))
               for x in (period,start,end)):
            raise ValueError('Integer period and frame boundaries required')
        if period<1 or period in periods or start<cursor or end<start or end>=n_frames:
            raise ValueError('Invalid, overlapping or repeated active period')
        if start>cursor:
            blocks.append((0,cursor,start))
        blocks.append((period,start,end+1))
        cursor=end+1; periods.add(period)
    if not periods:
        raise ValueError('At least one active period is required')
    if cursor<n_frames:
        blocks.append((0,cursor,n_frames))
    return blocks


def export_labels(frame_ids, labels, output, n_frames, active_blocks):
    blocks=timeline_blocks(n_frames,active_blocks)
    ids=np.asarray(frame_ids); labels=np.asarray(labels)
    if ids.ndim!=1 or labels.shape!=ids.shape or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('Aligned one-dimensional integer frame IDs and labels required')
    if not len(ids) or np.any(ids<1) or np.any(ids>n_frames) or len(np.unique(ids))!=len(ids):
        raise ValueError('Empty, duplicate or out-of-range one-based ATD frame IDs')
    if not np.isin(labels,['H','A','D','X']).all():
        raise ValueError('Invalid possession label')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Submission exists; choose a new output path')
    full=np.full(n_frames,'X',dtype='<U1')
    full[ids-1]=labels
    for period,start,end in blocks:
        if period==0:
            full[start:end]='X'
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create prevents silently replacing an earlier candidate.
    with output.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['frame','label'])
        writer.writerows(enumerate(full))
    official=load_official('evaluation/possession_metrics.py')
    error=official.validate_submission(official.read_label_csv(output),n_frames)
    if error:
        raise ValueError('Official format validation failed: '+error)
    return {'n_frames':n_frames,'csv_sha256':digest_file(output),
            'validation':'official_format_pass'}


def run(config_path, data_dir, output):
    config_path=Path(config_path).resolve()
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if cfg.get('source')!='tactical':
        raise ValueError('Submission inference requires the tactical timeline')
    if cfg.get('metadata_team_policy')!='epts_first_team_is_home':
        raise ValueError('Explicit documented EPTS team policy required')
    blocks=timeline_blocks(cfg['n_frames'],cfg['active_blocks'])
    output=Path(output)
    manifest=output.with_name(output.name+'.manifest.json')
    if output.exists() or manifest.exists():
        raise FileExistsError('Output or manifest exists; choose a new output path')
    model_path=(config_path.parent/cfg['model']).resolve()
    if digest_file(model_path)!=cfg['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    kind=cfg.get('model_kind','numpy_logistic_regression')
    if kind=='numpy_logistic_regression':
        model=json.loads(model_path.read_text(encoding='utf-8'))
        if model.get('kind')!=kind or model.get('features')!=list(NAMES):
            raise ValueError('Unsupported model or feature contract')
    elif kind in TEMPORAL_KINDS:
        import torch
        from . import temporal
        temporal.configure(20261108,4)
        model=torch.load(model_path,map_location='cpu',weights_only=True)
        if model.get('kind')!=kind:
            raise ValueError('Unsupported temporal checkpoint')
        if kind in STATE_KINDS and len(model.get('selected_transition_costs',[]))!=2:
            raise ValueError('v4 checkpoint requires selected owner and dead/live costs')
        if kind==ENSEMBLE_KIND and len(model.get('members',[]))<2:
            raise ValueError('Ensemble bundle requires at least two members')
    else:
        raise ValueError('Unsupported model kind')
    root=Path(data_dir)/'tactical'
    raw=root/(cfg['stem']+'_atd.txt'); meta=root/(cfg['stem']+'_atd.xml')
    teams=[team.get('id') for team in ET.parse(meta).getroot().iter('Team')]
    if len(teams)!=2 or not all(teams) or teams[0]==teams[1]:
        raise ValueError('Expected two distinct EPTS teams in documented home/away order')
    initial_hashes={'raw':digest_file(raw),'metadata':digest_file(meta)}
    tag_config=None
    if 'smart_tag_candidate' in cfg:
        if kind!='pytorch_temporal_conv_v1':raise ValueError('Tag candidate requires temporal model')
        tag_config_path=(config_path.parent/cfg['smart_tag_candidate']).resolve()
        tag_path=root/(cfg['stem']+'_smart_tagging.xml')
        initial_hashes['tags']=digest_file(tag_path)
        initial_hashes['tag_config']=digest_file(tag_config_path)
        tag_config=json.loads(tag_config_path.read_text())
    kickoff_rule=bool(cfg.get('kickoff_dead_rule'))
    if kickoff_rule:
        if kind not in STATE_KINDS:raise ValueError('Kickoff rule requires the v4 temporal model')
        tag_path=root/(cfg['stem']+'_smart_tagging.xml')
        initial_hashes['tags']=digest_file(tag_path)
    obs=atd.convert(raw,meta,cfg['match_id'],teams[0],teams[1],*cfg['pitch_m'],blocks)
    if not np.array_equal(obs.frame_id,np.arange(1,cfg['n_frames']+1)):
        raise ValueError('Expected complete tactical ATD timeline; check source and frame count')
    if kind=='numpy_logistic_regression':
        owners=team_states(predict(obs,50),obs.sequence_id)
        selected=model['selected_decoding']
        labels=decode(owners,probability(model,extract(obs)),obs.sequence_id,
                      selected['probability_threshold'],selected['half_width_frames'])
    elif kind in STATE_KINDS:
        from . import state_decoder
        from .retrain_v4 import MODULES
        from .feature_recovery_experiment import infer as infer_features
        module=MODULES[model.get('feature_module','v4')]
        if kind==ENSEMBLE_KIND:
            # Mean of the members' team-swap-averaged probabilities, renormalised, decoded once
            # (the same arithmetic as run_bench.py, which produced the selection evidence).
            p=np.mean([infer_features(member,obs,module) for member in model['members']],axis=0)
            p=p/p.sum(axis=1,keepdims=True)
        else:
            p=infer_features(model,obs,module)
        owner_cost,dead_live_cost=model['selected_transition_costs']
        labels=state_decoder.decode(p,obs.sequence_id,owner_cost,dead_live_cost)
        # The submission config may set the filter; otherwise the checkpoint's own value applies.
        min_controlled=int(cfg.get('min_controlled_frames',model.get('min_controlled_frames',0)))
        if min_controlled:
            from .possession_convention import absorb_short_spells
            labels=absorb_short_spells(labels,obs.sequence_id,min_controlled)
        if kickoff_rule:
            from .smart_tagging import load as load_tags,kickoff_dead_mask
            labels=labels.copy()
            labels[kickoff_dead_mask(load_tags(tag_path),obs.frame_id-1,obs.period)]='D'
    else:
        from .temporal_experiment import infer
        from .smart_tagging import apply_candidate
        p=infer(model,obs)
        p=apply_candidate(p,obs,tag_path if tag_config is not None else None,meta,tag_config)
        labels=temporal.decode(p,obs.sequence_id,model['selected_transition_penalty'])
    # Detect input replacement during either streaming pass or inference.
    final_hashes={'raw':digest_file(raw),'metadata':digest_file(meta)}
    if tag_config is not None:
        final_hashes.update(tags=digest_file(tag_path),tag_config=digest_file(tag_config_path))
    if kickoff_rule:
        final_hashes['tags']=digest_file(tag_path)
    if initial_hashes!=final_hashes:
        raise ValueError('Tracking inputs changed during inference')
    result=export_labels(obs.frame_id,labels,output,cfg['n_frames'],cfg['active_blocks'])
    result.update({'config_sha256':digest_file(config_path),'model_sha256':digest_file(model_path),
                   'input_sha256':initial_hashes,'source_sha256':source_hashes(),
                   'runtime':{'python':platform.python_version(),'numpy':np.__version__},
                   'inputs_used':['tactical_atd.txt','tactical_atd.xml'],
                   'ground_truth_accessed':False,'video_accessed':False,
                   'status':'Frozen external pilot inference; not a measured competition score'})
    if tag_config is not None:
        result['inputs_used'].append('tactical_smart_tagging.xml')
        result['status']='Experimental tag-assisted inference; not selected; no measured competition score'
        result['smart_tag_candidate']=tag_config
    result['model_kind']=kind
    if kind in TEMPORAL_KINDS:
        result['runtime']['torch']=str(torch.__version__)
    if kind in STATE_KINDS:
        result['status']='External-trained v4 inference; not a measured competition score'
    if kind==ENSEMBLE_KIND:
        result['status']='External-trained ensemble inference; not a measured competition score'
        result['ensemble_members']=len(model['members'])
    if kickoff_rule:
        result['inputs_used'].append('tactical_smart_tagging.xml')
        result['kickoff_dead_rule']=True
    if kind in STATE_KINDS:
        result['min_controlled_frames']=min_controlled
    write_json(manifest,result)
    return result
