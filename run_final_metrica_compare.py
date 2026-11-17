"""Final v6 vs final v7 (the checkpoints that would ship) on Metrica sample games 1 and 2.

Neither model trained on these games. Truth is the event-derived label as given (no merging);
the shipped 1 s filter is applied to predictions only. Each model uses the decoder costs stored in
its own checkpoint. External data only.
"""
import json
from pathlib import Path
import numpy as np
import torch
from fcb_external import metrica_sample, state_decoder, temporal, target_like
from fcb_external.feature_recovery_experiment import infer
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.retrain_v4 import MODULES, changed as base_changed
from fcb_external.feature_recovery_experiment import seed_for
from fcb_external.schema import write_json
ROOT = Path(__file__).resolve().parent

def changed(obs, sc, role, fold):
    if sc.get('kind') == 'target_like':
        return target_like.mask(obs, sc['missing_fraction'], sc['slow_weight'], sc['line_weight'],
                                seed_for(role, fold, obs.match_id, sc['name']))
    return base_changed(obs, sc, role, fold)

cfg = json.loads((ROOT / 'configs/retrain_v7.json').read_text())
wanted = ('clean', 'irregular40', 'crowded40', 'mixed_stress', 'slow40', 'target_like40')
scs = [s for s in cfg['scenarios'] if s['name'] in wanted]
temporal.configure(20261116, 4)
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
models = {'v6': load(ROOT / 'models/temporal_v6.pt'), 'v7': load(ROOT / 'models/temporal_v7.pt')}
module = MODULES['v6']
matches = metrica_sample.load_all(ROOT / 'external-data/metrica_sample')
cells = []
for name, (obs, truth) in matches.items():
    seq = np.asarray(obs.sequence_id)
    for sc in scs:
        work = changed(obs, sc, 'validation', 'metrica_final')
        row = {'match': name, 'scenario': sc['name']}
        for tag, ck in models.items():
            p = infer(ck, work, module)
            o, d = ck['selected_transition_costs']
            pred = absorb_short_spells(state_decoder.decode(p, seq, o, d), seq, 25)
            m = evaluate_labels(truth, pred, seq)['metrics']
            row[tag] = float(m['spell_f1_mean']); row[tag + '_costs'] = [o, d]
        cells.append(row); print('scored', name, sc['name'], round(row['v6'], 3), round(row['v7'], 3), flush=True)
write_json(ROOT / 'reports/temporal_v7/final_metrica_compare.json', {'cells': cells})
print(f'\n{"scenario":<14}{"v6":>8}{"v7":>8}{"delta":>8}')
for sc in scs:
    r = [c for c in cells if c['scenario'] == sc['name']]
    a, b = np.mean([c['v6'] for c in r]), np.mean([c['v7'] for c in r])
    print(f'{sc["name"]:<14}{a:>8.4f}{b:>8.4f}{b-a:>+8.4f}')
masked = [c for c in cells if c['scenario'] != 'clean']
print('masked mean', np.mean([c['v6'] for c in masked]), np.mean([c['v7'] for c in masked]))
