"""External-only acquisition/conversion/baseline/evaluation commands."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .schema import Observations, Targets, digest_file, write_json
from .idsse import convert
from .baseline import predict_all, VARIANTS
from .metrics import evaluate_labels
from .xmlio import validate_xml


def directory_hashes(directory, pattern='*.npy'):
    return {p.name:digest_file(p) for p in sorted(Path(directory).glob(pattern))}


def source_hashes():
    return {p.name:digest_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    base = (config_path.parent / config.get('project_root','.')).resolve()
    paths = {key:(base/config[key]).resolve() for key in ('metadata','positions','events')}
    output = (base/config['output']).resolve()
    if config.get('coordinate_system') != 'dfl-centred-metres':
        raise ValueError('Only documented DFL centred-metre coordinates are supported')
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Missing external input: {path}')
        if 'BAR-ATH' in path.name or 'more-than-a-hack-2027-dataset' in path.parts:
            raise ValueError('Restricted competition data is not permitted in this milestone')
    hashes = {key:digest_file(path) for key,path in paths.items()}
    for key,expected in config.get('expected_sha256',{}).items():
        if hashes.get(key) != expected:
            raise ValueError(f'External {key} checksum does not match the pinned source')
    # Check event XML structure only; event annotations never feed observations or inference.
    validate_xml(paths['events'],config['match_id'],'Event')
    obs,targets,audit = convert(paths['positions'],paths['metadata'],config['match_id'])
    if not audit['ball_rows'] or audit['status_domains'].keys()=={'MISSING'} or \
            audit['possession_domains'].keys()=={'MISSING'} or not targets.valid.any():
        raise ValueError('External possession/status targets absent or entirely unusable; no guesses made')
    obs.save(output/'observations')
    targets.save(output/'targets',obs)
    manifest = {'schema_version':1,'match_id':obs.match_id,
        'purpose':'External development fixture, not held-out testing',
        'source_revision':config.get('source_revision','unrecorded'),
        'source_urls':config.get('source_urls',[]),
        'input_sha256':hashes,'input_size_bytes':{k:p.stat().st_size for k,p in paths.items()},
        'source_sha256':source_hashes(),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'timeline_sha256':obs.timeline_hash(),
        'observation_sha256':directory_hashes(output/'observations'),
        'target_sha256':directory_hashes(output/'targets'),
        'label_mapping_version':1,'fps':obs.fps,
        'sequence_policy':'Break on period change or missing observation frame ID; never use target values',
        'licence':'CC BY 4.0; credit DFL and Bassek et al. (2025); conversion by FCB Hackathon team',
        'audit':audit}
    write_json(output/'manifest.json',manifest)
    print(f'Prepared external match {obs.match_id}; report: {output / "manifest.json"}')


def baseline(observation_dir, out):
    obs = Observations.load(observation_dir)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for name,labels in predict_all(obs).items():
        np.save(out/(name+'.npy'),labels,allow_pickle=False)
    write_json(out/'manifest.json',{'timeline_sha256':obs.timeline_hash(),
        'observation_sha256':directory_hashes(observation_dir),
        'variants_half_width_frames':VARIANTS,'fps':25,
        'policy':'H/A-only votes, H ties, reset per observation sequence; no target inputs',
        'prediction_sha256':directory_hashes(out),'source_sha256':source_hashes()})
    print(f'Generated four fixed external baselines in {out}')


def evaluate(observation_dir, target_dir, prediction_dir, out, official_repo=None):
    obs = Observations.load(observation_dir)
    targets = Targets.load(target_dir,obs)
    prediction_dir,out = Path(prediction_dir),Path(out)
    manifest = json.loads((prediction_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['timeline_sha256'] != obs.timeline_hash() or \
            manifest['observation_sha256'] != directory_hashes(observation_dir):
        raise ValueError('Predictions came from different observation content or timeline')
    if manifest['prediction_sha256'] != directory_hashes(prediction_dir):
        raise ValueError('Prediction checksum mismatch')
    results = {}
    for name in VARIANTS:
        labels = np.load(prediction_dir/(name+'.npy'),allow_pickle=False)
        if labels.shape != targets.label.shape:
            raise ValueError('Prediction shape mismatch')
        results[name] = evaluate_labels(targets.label,labels,obs.sequence_id,official_repo)
    report = {'development_match':obs.match_id,'variants':results,
        'purpose':'Fixed external reference benchmark. No model/window selection and no tuning.',
        'observation_sha256':directory_hashes(observation_dir),
        'target_sha256':directory_hashes(target_dir),
        'prediction_sha256':directory_hashes(prediction_dir),
        'target_coverage_fraction':float(targets.valid.mean()),
        'sequence_count':int(obs.sequence_id[-1])+1,
        'label_counts':dict(Counter(targets.label.tolist()))}
    write_json(out/'benchmark.json',report)
    columns = ('variant','macroF1','spellF1','gt_possessions','pred_possessions','mean_duration_s')
    with (out/'scoreboard.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle,lineterminator='\n'); writer.writerow(columns)
        for name,result in results.items():
            m=result['metrics']
            writer.writerow([name,m['macro_f1'],m['spell_f1_mean'],m['n_gt_spells'],
                             m['n_pred_spells'],m['mean_spell_duration_s']])
    print(f'External development benchmark written to {out}; no competition score inferred.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--config',type=Path,required=True)
    b=sub.add_parser('baseline'); b.add_argument('--observations',type=Path,required=True)
    b.add_argument('--out',type=Path,required=True)
    e=sub.add_parser('evaluate'); e.add_argument('--observations',type=Path,required=True)
    e.add_argument('--targets',type=Path,required=True); e.add_argument('--predictions',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True); e.add_argument('--official-repo',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            prepare(args.config)
        elif args.command=='baseline':
            baseline(args.observations,args.out)
        else:
            evaluate(args.observations,args.targets,args.predictions,args.out,args.official_repo)
    except (ValueError,OSError,ET.ParseError,KeyError,TypeError) as exc:
        print(f'{args.command} failed: {exc}',file=sys.stderr)
        return 1
    return 0
