"""Run locally: detailed match-derived audit stays in the requested local output."""
import argparse
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from fcb_external import atd, smart_tagging
from fcb_external.submission import timeline_blocks
from fcb_external.schema import write_json, digest_file


def run(data_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new local report path')
    report = {
        'status': 'incomplete', 'audit_executed': False, 'contract_passed': False,
        'ground_truth_accessed': False, 'video_accessed': False,
        'required_checks': {
            'resolved_team_mapping': False, 'complete_timeline': False,
            'tag_bounds': False, 'input_immutability': False,
        },
    }
    checks = report['required_checks']
    try:
        cfg = json.loads((Path(__file__).parent / 'configs/submission_selected.json').read_text())
        folder = Path(data_dir) / 'tactical'
        stem = cfg['stem']
        paths = {k: folder / (stem + suffix) for k, suffix in [
            ('tags', '_smart_tagging.xml'), ('metadata', '_atd.xml'), ('tracking', '_atd.txt')]}
        hashes = {k: digest_file(p) for k, p in paths.items()}
        report['source_sha256'] = hashes
        tags = smart_tagging.load(paths['tags'])
        report['tag_count'] = len(tags)
        try:
            team_map = smart_tagging.metadata_team_map(paths['metadata'])
            checks['resolved_team_mapping'] = True
            report['team_mapping_status'] = 'resolved'
        except ValueError:
            # Continue diagnostics with neutral evidence, but fail the required mapping check.
            team_map = {}
            report['team_mapping_status'] = 'unresolved; explicit metadata name mapping required'
            report['team_mapping_error'] = traceback.format_exc()
        ids = [t.get('id') for t in ET.parse(paths['metadata']).getroot().iter('Team')]
        if len(ids) != 2:
            raise ValueError('Two EPTS teams required')
        obs = atd.convert(paths['tracking'], paths['metadata'], cfg['match_id'], ids[0], ids[1],
                          *cfg['pitch_m'], timeline_blocks(cfg['n_frames'], cfg['active_blocks']))
        frames = obs.frame_id - 1
        evidence = smart_tagging.owner_evidence(tags, frames, obs.period, team_map)
        checks['complete_timeline'] = bool(np.array_equal(obs.frame_id, np.arange(1, cfg['n_frames'] + 1)))
        out_of_range = sum(not (0 <= t.start_frame < t.end_frame <= cfg['n_frames']) for t in tags)
        checks['tag_bounds'] = out_of_range == 0
        report.update(
            team_map=team_map, complete_timeline=checks['complete_timeline'],
            unknown_tag_teams=sorted({t.team for t in tags if t.team and t.team not in team_map}),
            conflicting_tags=sum(bool(t.conflicts) for t in tags),
            out_of_range_intervals=out_of_range,
            missing_ball_frames=int((~obs.ball_observed).sum()),
            missing_ball_with_phase_evidence=int(((~obs.ball_observed) & (evidence >= 0)).sum()),
            limitations='Operational contract audit only; no possession-accuracy measurement. Phase windows are not possessions.')
        # Automatic kickoff labels are noisy: containment is diagnostic, never a readiness gate.
        report['kickoff_containment'] = {
            str(half): any(t.code == 'set pieces' and t.kind == 'kick off (start)' and
                           t.half == label and t.start_frame <= frame < t.end_frame for t in tags)
            for half, label, frame in [(1, '1st half', 11185), (2, '2nd half', 113742)]}
        checks['input_immutability'] = hashes == {k: digest_file(p) for k, p in paths.items()}
        report['audit_executed'] = True
        report['contract_passed'] = all(checks.values())
        report['status'] = 'complete' if report['contract_passed'] else 'needs_local_review'
    except Exception:
        report['status'] = 'needs_local_review'
        report['local_error'] = traceback.format_exc()
    write_json(output, report)
    # Only operational status leaves the local report; no match-derived details.
    print('Local audit status: ' + report['status'] + '. Detailed report stays local.')
    return report['contract_passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.data_dir, args.out) else 1)
