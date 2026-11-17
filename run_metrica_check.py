"""Score the released pipelines on Metrica's public sample matches (external data only).

Purpose: the sample matches are the only public data labelled under the organiser's own event
definitions. This compares v2, v4 and v5 on them, with the ball masked the same way as in
training, and sweeps the controlled-possession filter so its duration is chosen on external
matches of the same provider. No competition data is read.

    python run_metrica_check.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, state_decoder, temporal, team_features_v4
from fcb_external.feature_recovery_experiment import infer, seed_for
from fcb_external.temporal_experiment import infer as infer_v2
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells, runs
from fcb_external.robustness import corrupt
from fcb_external.schema import write_json

ROOT=Path(__file__).resolve().parent
FILTERS=(0,25,50)
TRUTH_FILTERS=(25,50)


def spell_stats(label,seq):
    lengths=[b-a for a,b in runs(label,seq) if label[a] in ('H','A')]
    return {'spells':len(lengths),'median_s':float(np.median(lengths))/25,'mean_s':float(np.mean(lengths))/25,
            'under_1s':int(np.sum(np.array(lengths)<25)),'under_2s':int(np.sum(np.array(lengths)<50)),
            'dead_fraction':float(np.mean(label=='D'))}


def main():
    from fcb_external.retrain_v4 import MODULES, changed
    cfg=json.loads((ROOT/'configs/retrain_v6.json').read_text())
    scenarios=[s for s in cfg['scenarios'] if s['name'] in ('clean','irregular40','crowded40','mixed_stress','slow40')]
    temporal.configure(20261115,4)
    load=lambda name:torch.load(ROOT/'models'/name,map_location='cpu',weights_only=True)
    v2=load('temporal_v2.pt')
    models={key:load(f'temporal_{key}.pt') for key in ('v4','v5','v6') if (ROOT/'models'/f'temporal_{key}.pt').exists()}
    matches=metrica_sample.load_all(ROOT/'external-data/metrica_sample')
    report={'truth':{},'cells':[]}
    for name,(obs,truth) in matches.items():
        seq=np.asarray(obs.sequence_id)
        report['truth'][name]={str(f):spell_stats(absorb_short_spells(truth,seq,f),seq) for f in (0,25,50)}
        print(name,report['truth'][name],flush=True)
        for scenario in scenarios:
            work=changed(obs,scenario,'validation','metrica')
            rows={'v2':temporal.decode(infer_v2(v2,work),seq,v2['selected_transition_penalty'])}
            for key,model in models.items():
                owner,dead=model['selected_transition_costs']
                module=MODULES[model.get('feature_module','v4')]
                decoded=state_decoder.decode(infer(model,work,module),seq,owner,dead)
                for frames in FILTERS:
                    rows[f'{key}+filter{frames}']=absorb_short_spells(decoded,seq,frames)
            for pipeline,predicted in rows.items():
                m=evaluate_labels(truth,predicted,seq)['metrics']
                cell={'match':name,'scenario':scenario['name'],'pipeline':pipeline,
                    'spellF1':float(m['spell_f1_mean']),'macroF1':float(m['macro_f1']),
                    'pred_spells':int(m['n_pred_spells']),'gt_spells':int(m['n_gt_spells'])}
                # The same prediction against stricter readings of 'controlled possession'.
                for frames in TRUTH_FILTERS:
                    strict=absorb_short_spells(truth,seq,frames)
                    cell[f'spellF1_truth{frames}']=float(evaluate_labels(strict,predicted,seq)['metrics']['spell_f1_mean'])
                report['cells'].append(cell)
            print('scored',name,scenario['name'],flush=True)
    masked=[c for c in report['cells'] if c['scenario']!='clean']
    table={}
    for group,cells in (('masked',masked),('clean',[c for c in report['cells'] if c['scenario']=='clean'])):
        for pipeline in dict.fromkeys(c['pipeline'] for c in cells):
            mine=[c for c in cells if c['pipeline']==pipeline]
            per_match=[np.mean([c['spellF1'] for c in mine if c['match']==m]) for m in matches]
            table.setdefault(pipeline,{})[group]={'spellF1':float(np.mean(per_match)),
                'per_match':[round(float(v),4) for v in per_match],
                'pred_spells':float(np.mean([c['pred_spells'] for c in mine])),
                'gt_spells':float(np.mean([c['gt_spells'] for c in mine]))}
    report['summary']=table
    write_json(ROOT/'reports/metrica_check/results.json',report)
    for pipeline in table:
        mine=[c for c in masked if c['pipeline']==pipeline]
        for frames in TRUTH_FILTERS:
            table[pipeline]['masked'][f'truth{frames}']=float(np.mean([np.mean([c[f'spellF1_truth{frames}'] for c in mine if c['match']==m]) for m in matches]))
    write_json(ROOT/'reports/metrica_check/results.json',report)
    print('masked spellF1 against event labels as given, and with opposing spells under 1 s / 2 s absorbed:')
    for pipeline,row in table.items():
        print(f'{pipeline:<16}{row["masked"]["spellF1"]:>8.4f}{row["masked"]["truth25"]:>8.4f}{row["masked"]["truth50"]:>8.4f}')
    print(f'{"pipeline":<16}{"masked":>8}{"clean":>8}{"pred":>7}{"truth":>7}   per-match masked')
    for pipeline,row in table.items():
        print(f'{pipeline:<16}{row["masked"]["spellF1"]:>8.4f}{row["clean"]["spellF1"]:>8.4f}'
              f'{row["masked"]["pred_spells"]:>7.0f}{row["masked"]["gt_spells"]:>7.0f}   {row["masked"]["per_match"]}')
    print('per scenario (spellF1 / macroF1), pipelines with the 1 s filter:')
    for pipeline in table:
        if pipeline.endswith('filter25') or pipeline=='v2':
            parts=[]
            for scenario in scenarios:
                mine=[c for c in report['cells'] if c['pipeline']==pipeline and c['scenario']==scenario['name']]
                parts.append(f'{scenario["name"]} {np.mean([c["spellF1"] for c in mine]):.3f}/{np.mean([c["macroF1"] for c in mine]):.3f}')
            print(f'{pipeline:<16}'+'  '.join(parts))


if __name__=='__main__':
    main()
