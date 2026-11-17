"""Paired comparison of v7 and v6 fold models on the same held-out DFL matches and masks.

For each fold, both models see exactly the same corrupted observations (same seeds). v6 uses
its selected costs (5, 10); v7 uses the costs its own CV selected. Both get the same 1 s
controlled-possession filter. External data only; no competition data is read.

    python run_compare_v7_v6.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import state_decoder, temporal, target_like
from fcb_external.feature_recovery_experiment import infer, seed_for
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.retrain_v4 import MODULES, load_external, changed as base_changed
from fcb_external.schema import write_json

ROOT = Path(__file__).resolve().parent
FILTER = 25


def changed(obs, scenario, role, fold):
    if scenario.get('kind') == 'target_like':
        return target_like.mask(obs, scenario['missing_fraction'], scenario['slow_weight'],
                                scenario['line_weight'], seed_for(role, fold, obs.match_id, scenario['name']))
    return base_changed(obs, scenario, role, fold)


def main():
    cfg = json.loads((ROOT / 'configs/retrain_v7.json').read_text())
    decision = json.loads((ROOT / 'reports/temporal_v7/decision.json').read_text())
    costs7 = tuple(decision['selected_costs'])
    costs6 = (5.0, 10.0)
    wanted = ('clean', 'irregular40', 'crowded40', 'mixed_stress', 'slow40', 'target_like40')
    scenarios = [s for s in cfg['scenarios'] if s['name'] in wanted]
    temporal.configure(20261116, 4)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    module = MODULES['v6']
    names = np.array(['H', 'A', 'D'])
    cells = []
    for fold in cfg['folds']:
        m6 = load(ROOT / f"models/temporal_v6/{fold['name']}.pt")
        m7 = load(ROOT / f"models/temporal_v7/{fold['name']}.pt")
        for match in fold['validation']:
            obs, targets = load_external(ROOT, match)
            truth = np.asarray(targets.label)
            seq = np.asarray(obs.sequence_id)
            for sc in scenarios:
                # v7 seeds are keyed by its own fold seeds only through the fold name, so both models share masks.
                work = changed(obs, sc, 'validation', fold['name'])
                row = {'fold': fold['name'], 'match': match, 'scenario': sc['name'],
                       'missing': float((~work.ball_observed).mean())}
                for tag, model, costs in (('v6', m6, costs6), ('v7', m7, costs7)):
                    d = state_decoder.decode(infer(model, work, module), seq, *costs)
                    pred = absorb_short_spells(d, seq, FILTER)
                    mm = evaluate_labels(truth, pred, seq)['metrics']
                    row[tag] = float(mm['spell_f1_mean']); row[tag + '_macro'] = float(mm['macro_f1'])
                    row[tag + '_spells'] = int(mm['n_pred_spells']); row['gt_spells'] = int(mm['n_gt_spells'])
                cells.append(row)
                print('scored', fold['name'], match, sc['name'], round(row['v6'], 3), round(row['v7'], 3), flush=True)
    write_json(ROOT / 'reports/temporal_v7/compare_v6_rawtruth.json', {'costs_v6': costs6, 'costs_v7': costs7, 'cells': cells})
    print(f'\n{"scenario":<14}{"v6":>8}{"v7":>8}{"delta":>8}  (mean over the 7 held-out matches)')
    for sc in scenarios:
        rows = [c for c in cells if c['scenario'] == sc['name']]
        a, b = np.mean([c['v6'] for c in rows]), np.mean([c['v7'] for c in rows])
        wins = sum(c['v7'] > c['v6'] for c in rows)
        print(f'{sc["name"]:<14}{a:>8.4f}{b:>8.4f}{b-a:>+8.4f}  v7 better on {wins}/{len(rows)} matches')


if __name__ == '__main__':
    main()
