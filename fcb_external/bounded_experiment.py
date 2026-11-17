"""Frozen scenario selection and provenance for bounded development experiments."""
import json
from pathlib import Path

from .cli import source_hashes
from .experiment import input_fingerprint
from .schema import digest_file


# Original four robustness_v2 definitions, independent of config list order.
_SCENARIOS_JSON = '''[
  {"name":"clean","seed":100},
  {"name":"short_occlusion","seed":101,"ball_gap_frames":25,"ball_gap_period_frames":250},
  {"name":"long_occlusion","seed":102,"ball_gap_frames":125,"ball_gap_period_frames":500},
  {"name":"mixed_noise","seed":103,"ball_gap_frames":50,"ball_gap_period_frames":250,"player_drop_probability":0.1,"coordinate_noise_std_m":0.5,"permute_slots_within_team":true}
]'''


def frozen_scenarios(config):
    """Reject changed/missing/duplicate frozen definitions; return fresh copies."""
    frozen = json.loads(_SCENARIOS_JSON)
    for expected in frozen:
        matches = [row for row in config['scenarios'] if row.get('name') == expected['name']]
        if len(matches) != 1 or json.dumps(matches[0], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f'Frozen scenario definition mismatch: {expected["name"]}')
    return frozen


def capture_provenance(root, runner, model, match):
    """Fingerprint the package, runner, configs, checkpoint and prepared inputs."""
    root = Path(root)
    sources = source_hashes()
    sources[str(runner)] = digest_file(root / runner)
    return {
        'source_sha256': sources,
        'config_sha256': {name: digest_file(root / 'configs' / name)
                          for name in ('experiment_v1.json', 'robustness_v2.json')},
        'model_sha256': digest_file(root / model),
        'external_match': match,
        'external_inputs': {kind: input_fingerprint(root / 'artifacts' / match / kind)
                            for kind in ('observations', 'targets')},
    }


def verify_provenance(expected, root, runner, model, match):
    """Fail closed if any recorded source or input changed during execution."""
    if capture_provenance(root, runner, model, match) != expected:
        raise ValueError('Experiment provenance changed during execution')
