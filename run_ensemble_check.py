"""Does averaging the v6 fold checkpoints with the final v6 model help? External data only.

Metrica sample games 1 and 2 are in no model's training set, so this is a fair comparison.
Probabilities are averaged (each model keeps its own scaling and team-swap averaging), then
decoded once with the existing v6 costs. No costs are re-tuned. No competition data is read.

    python run_ensemble_check.py
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, state_decoder, temporal
from fcb_external.feature_recovery_experiment import infer
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.retrain_v4 import MODULES, changed
from fcb_external.schema import write_json

ROOT = Path(__file__).resolve().parent
COSTS = (5.0, 10.0)
FILTER = 25


def main():
    cfg = json.loads((ROOT / 'configs/retrain_v6.json').read_text())
    scenarios = [s for s in cfg['scenarios'] if s['name'] in ('clean', 'irregular40', 'crowded40', 'mixed_stress', 'slow40')]
    temporal.configure(20261115, 4)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    final = load(ROOT / 'models/temporal_v6.pt')
    folds = {k: load(ROOT / f'models/temporal_v6/{k}.pt') for k in 'ABCD'}
    module = MODULES['v6']
    matches = metrica_sample.load_all(ROOT / 'external-data/metrica_sample')
    cells = []
    for name, (obs, truth) in matches.items():
        seq = np.asarray(obs.sequence_id)
        for scenario in scenarios:
            work = changed(obs, scenario, 'validation', 'metrica')
            probs = {'final': infer(final, work, module)}
            for k, m in folds.items():
                probs[k] = infer(m, work, module)
            combos = {
                'final': ['final'],
                'folds4': list('ABCD'),
                'final+folds4': ['final'] + list('ABCD'),
                'A': ['A'], 'B': ['B'], 'C': ['C'], 'D': ['D'],
            }
            for combo, keys in combos.items():
                p = np.mean([probs[k] for k in keys], axis=0)
                decoded = state_decoder.decode(p, seq, *COSTS)
                predicted = absorb_short_spells(decoded, seq, FILTER)
                m = evaluate_labels(truth, predicted, seq)['metrics']
                cells.append({'match': name, 'scenario': scenario['name'], 'combo': combo,
                              'spellF1': float(m['spell_f1_mean']), 'macroF1': float(m['macro_f1']),
                              'pred_spells': int(m['n_pred_spells']), 'gt_spells': int(m['n_gt_spells'])})
            print('scored', name, scenario['name'], flush=True)
    write_json(ROOT / 'reports/metrica_check/ensemble_results.json', {'costs': COSTS, 'filter': FILTER, 'cells': cells})
    combos = list(dict.fromkeys(c['combo'] for c in cells))
    print(f'{"combo":<14}{"masked":>8}{"clean":>8}   per-scenario spellF1 (mean of 2 games)')
    for combo in combos:
        def mean(group):
            return np.mean([np.mean([c['spellF1'] for c in cells if c['combo'] == combo and c['match'] == m and group(c)])
                            for m in matches])
        per = '  '.join(f'{s["name"]} {mean(lambda c, n=s["name"]: c["scenario"] == n):.3f}' for s in scenarios)
        print(f'{combo:<14}{mean(lambda c: c["scenario"] != "clean"):>8.4f}{mean(lambda c: c["scenario"] == "clean"):>8.4f}   {per}')


if __name__ == '__main__':
    main()
