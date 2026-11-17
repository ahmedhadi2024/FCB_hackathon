"""Gradient-free weight edit on the dead-ball output weights of the v7 fold networks.

Read first: for each fold network, score every last-layer channel by how much it separates dead-ball
frames from live frames on that fold's TRAINING match, times its output weight into the D logit.
Edit: multiply the D-row output weights of the top-k channels by a gain g. No training.

Protocol (external data only, no competition data and no public-window labels):
  1. Choose (k, g) on fold A's held-out matches (J03WN1, J03WR9).
  2. Apply the same recipe unchanged to folds B, C and D and score their held-out matches.
  Keep the edit only if it wins on the confirmation folds, not just on fold A.

    python run_weight_edit.py select      # grid on fold A
    python run_weight_edit.py confirm K G # recipe on folds B, C, D
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from fcb_external import state_decoder, temporal, target_like
from fcb_external.feature_recovery_experiment import seed_for
from fcb_external.metrics import evaluate_labels
from fcb_external.possession_convention import absorb_short_spells
from fcb_external.retrain_v4 import MODULES, load_external, changed as base_changed
from fcb_external.schema import write_json

ROOT = Path(__file__).resolve().parent
FILTER = 25
COSTS = (5.0, 30.0)
SCEN = ('clean', 'irregular40', 'slow40', 'mixed_stress', 'target_like40')


def changed(obs, sc, role, fold):
    if sc.get('kind') == 'target_like':
        return target_like.mask(obs, sc['missing_fraction'], sc['slow_weight'], sc['line_weight'],
                                seed_for(role, fold, obs.match_id, sc['name']))
    return base_changed(obs, sc, role, fold)


def make_net(ck):
    arch = ck['network']
    net = temporal.TemporalNet(len(ck['features']), arch['channels'], arch['dilations'])
    net.load_state_dict(ck['state_dict']); net.eval()
    return net


def hidden_and_labels(net, ck, obs, y, module):
    """Last-layer activations on one training match under target-like ball loss."""
    x = module.extract(obs)
    x = (x - ck['mean'].numpy()) / ck['scale'].numpy()
    seq = np.asarray(obs.sequence_id)
    acts = []
    with torch.inference_mode():
        for lo in range(0, len(x), 8192):
            hi = min(len(x), lo + 8192)
            left, right = max(0, lo - net.radius), min(len(x), hi + net.radius)
            t = torch.from_numpy(np.ascontiguousarray(x[left:right].T))[None]
            h = torch.relu(net.input(t))
            for b in net.blocks:
                h = torch.relu(h + b(h))
            acts.append(h[0].T.numpy()[lo - left:hi - left])
    return np.concatenate(acts)


def channel_scores(net, acts, labels):
    dead = labels == 'D'
    diff = acts[dead].mean(0) - acts[~dead & np.isin(labels, ['H', 'A'])].mean(0)
    w_d = net.output.weight[2, :, 0].detach().numpy()
    return diff * w_d                       # positive: channel helps tell dead from live through w_D


def edited(ck, scores, k, g):
    new = {kk: v.clone() for kk, v in ck['state_dict'].items()}
    top = np.argsort(-scores)[:k]
    new['output.weight'][2, torch.from_numpy(top).long(), 0] *= g
    out = dict(ck); out['state_dict'] = new
    return out, top


def probs(net, ck, x, seq, module):
    mean, scale = ck['mean'].numpy(), ck['scale'].numpy()
    a = temporal.probabilities(net, (x - mean) / scale, seq)
    b = temporal.probabilities(net, (module.swap(x) - mean) / scale, seq)
    return (a + b[:, [1, 0, 2]]) * .5


def score(p, truth, seq):
    d = state_decoder.decode(p, seq, *COSTS)
    return evaluate_labels(truth, absorb_short_spells(d, seq, FILTER), seq)['metrics']


def run_fold(cfg, fold, configs):
    module = MODULES['v6']
    ck = torch.load(ROOT / f"models/temporal_v7/{fold['name']}.pt", map_location='cpu', weights_only=True)
    base = make_net(ck)
    scenarios = [s for s in cfg['scenarios'] if s['name'] in SCEN]
    tr_match = fold['train'][0]
    tr_obs, tr_t = load_external(ROOT, tr_match)
    tr_sc = next(s for s in cfg['scenarios'] if s['name'] == 'target_like40')
    acts = hidden_and_labels(base, ck, changed(tr_obs, tr_sc, 'attribution', fold['name']), None, module)
    scores = channel_scores(base, acts, np.asarray(tr_t.label))
    nets = {'base': (base, ck)}
    tops = {}
    for k, g in configs:
        ek, top = edited(ck, scores, k, g)
        nets[f'k{k}_g{g}'] = (make_net(ek), ek); tops[f'k{k}_g{g}'] = top.tolist()
    rows = []
    for match in fold['validation']:
        obs, targets = load_external(ROOT, match)
        seq = np.asarray(obs.sequence_id)
        truth = absorb_short_spells(np.asarray(targets.label), seq, FILTER)
        for sc in scenarios:
            work = changed(obs, sc, 'validation', fold['name'])
            x = module.extract(work)
            for name, (net, c) in nets.items():
                m = score(probs(net, c, x, seq, module), truth, seq)
                rows.append({'fold': fold['name'], 'match': match, 'scenario': sc['name'], 'config': name,
                             'spellF1': float(m['spell_f1_mean']), 'macroF1': float(m['macro_f1'])})
            print('scored', fold['name'], match, sc['name'], flush=True)
    return rows, tops


def table(rows):
    configs = list(dict.fromkeys(r['config'] for r in rows))
    print(f'{"config":<12}' + ''.join(f'{s:>14}' for s in SCEN) + f'{"masked":>9}')
    base = None
    for c in configs:
        per = [np.mean([r['spellF1'] for r in rows if r['config'] == c and r['scenario'] == s]) for s in SCEN]
        masked = np.mean([v for s, v in zip(SCEN, per) if s != 'clean'])
        if c == 'base':
            base = per
        d = '' if base is None or c == 'base' else '  delta ' + ' '.join(f'{a-b:+.4f}' for a, b in zip(per, base))
        print(f'{c:<12}' + ''.join(f'{v:>14.4f}' for v in per) + f'{masked:>9.4f}{d}')


if __name__ == '__main__':
    cfg = json.loads((ROOT / 'configs/retrain_v7.json').read_text())
    temporal.configure(20261116, 4)
    folds = {f['name']: f for f in cfg['folds']}
    if sys.argv[1] == 'select':
        grid = [(k, g) for k in (4, 8, 16) for g in (1.25, 1.5, 2.0)]
        rows, tops = run_fold(cfg, folds['A'], grid)
        write_json(ROOT / 'reports/temporal_v7/weight_edit_select.json', {'rows': rows, 'tops': tops})
        table(rows)
    else:
        k, g = int(sys.argv[2]), float(sys.argv[3])
        rows = []
        for name in 'BCD':
            r, _ = run_fold(cfg, folds[name], [(k, g)]); rows += r
        write_json(ROOT / 'reports/temporal_v7/weight_edit_confirm.json', {'k': k, 'g': g, 'rows': rows})
        table(rows)
