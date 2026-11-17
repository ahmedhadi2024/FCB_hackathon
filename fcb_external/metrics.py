"""Thin adapter around the UNMODIFIED public official metric implementation."""
import importlib.util
from pathlib import Path
import numpy as np
from .schema import digest_file

OFFICIAL = Path(__file__).resolve().parents[1] / 'vendor' / 'official'


def load_official(relative, official_repo=None):
    path = (Path(official_repo) if official_repo else OFFICIAL) / relative
    if not path.is_file():
        raise ValueError(f'Public official source not found: {path}')
    spec = importlib.util.spec_from_file_location('fcb_official_'+path.stem,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_labels(gt, pred, sequence_id, official_repo=None):
    if len(gt)!=len(pred) or len(gt)!=len(sequence_id) or not len(gt):
        raise ValueError('Nonempty, equally aligned evaluation arrays required')
    pm = load_official('evaluation/possession_metrics.py',official_repo)
    raw_gt,raw_pred = pm.label_codes(gt),pm.label_codes(pred)
    boundaries = np.flatnonzero(np.diff(sequence_id))+1
    # np.insert uses positions in the original arrays; X splits every temporal valid run.
    g = np.insert(raw_gt,boundaries,pm.X)
    p = np.insert(raw_pred,boundaries,pm.X)
    result = pm.evaluate(g,p,bootstrap_n=0)
    cm = pm.scored_confusion(pm.confusion(g,p))
    per_class = {}
    for i,label in enumerate(('H','A','D')):
        precision,recall,f1 = pm.per_class_prf(cm,i)
        per_class[label] = {'precision':precision,'recall':recall,'f1':f1,
                            'support':int(cm[i].sum())}
    path = (Path(official_repo) if official_repo else OFFICIAL) / 'evaluation/possession_metrics.py'
    spells = pm.spell_scores(g,pm.fill_abstentions(p,pm.valid_runs(g)),pm.valid_runs(g))
    return {'metrics':result,'per_class':per_class,
            'spell_f1_by_iou':{str(k):v['f1'] for k,v in spells['per_iou'].items()},
            'observation_rows':len(gt),'graded_frames':int(np.sum(raw_gt!=pm.X)),
            'unknown_target_frames':int(np.sum(raw_gt==pm.X)),
            'evaluation_separator_rows':len(boundaries),
            'metric_source_sha256':digest_file(path),'fps':25,
            'interpretation':'External development match only; not held-out or competition performance.'}
