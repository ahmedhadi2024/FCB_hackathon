"""v7 retraining: v6 pipeline plus the target-like ball-loss scenario. External data only.

    python run_retrain_v7.py cv|final
"""
import sys
from pathlib import Path
from fcb_external import retrain_v4, target_like
from fcb_external.feature_recovery_experiment import seed_for

_orig = retrain_v4.changed

def changed(obs, scenario, role, fold_name):
    if scenario.get('kind') == 'target_like':
        seed = seed_for(role, fold_name, obs.match_id, scenario['name'])
        return target_like.mask(obs, scenario['missing_fraction'], scenario['slow_weight'],
                                scenario['line_weight'], seed)
    return _orig(obs, scenario, role, fold_name)

retrain_v4.changed = changed

if __name__ == '__main__':
    stage = sys.argv[1]
    root = Path(__file__).resolve().parent
    (retrain_v4.run_cv if stage == 'cv' else retrain_v4.run_final)(root, 'retrain_v7.json')
