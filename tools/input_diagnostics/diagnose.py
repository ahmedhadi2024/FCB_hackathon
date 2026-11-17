"""Local input-quality audit. No labels, video reads, training or network access."""

import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def position(group):
    """Missing or nonfinite x/y is unobserved; speed is not required."""
    try:
        fields = group.split(',')
        x, y = float(fields[0]), float(fields[1])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (ValueError, IndexError):
        return None


def missing_runs(observed, fps):
    lengths = []
    run = 0
    for value in observed:
        if not value:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return {'count': len(lengths), 'max_frames': max(lengths, default=0),
            'max_seconds': max(lengths, default=0) / fps,
            'length_histogram_frames': dict(sorted(Counter(lengths).items()))}


def audit_tracking(raw_path, metadata_path, expected_frames, fps):
    root = ET.parse(metadata_path).getroot()
    teams = [item.get('id') for item in root.iter('Team')]
    slots = [item.get('teamId') for item in root.iter('Player')]
    if len(teams) != 2 or len(set(teams)) != 2 or None in teams or not slots:
        raise ValueError('Expected two distinct teams and at least one track slot')
    if any(team not in teams for team in slots):
        raise ValueError('Track slots refer to an unknown team')
    slot_teams = [teams.index(team) for team in slots]
    seen, ball_seen = bytearray(expected_frames), bytearray(expected_frames)
    player_counts = Counter()
    team_counts = [0, 0]
    counters = Counter()
    previous = None
    with Path(raw_path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            counters['rows'] += 1
            parts = line.strip().split(':')
            try:
                frame = int(parts[0]) - 1
                if len(parts) != 3:
                    raise ValueError()
            except ValueError:
                counters['malformed_rows'] += 1
                continue
            if not 0 <= frame < expected_frames:
                counters['out_of_range_ids'] += 1
                continue
            if previous is not None and frame != previous + 1:
                counters['nonsequential_rows'] += 1
            previous = frame
            if seen[frame]:
                counters['duplicate_frame_ids'] += 1
                continue  # First occurrence wins; duplicates are still a validation failure.
            seen[frame] = 1
            groups = parts[1].split(';')
            if len(groups) != len(slots):
                counters['slot_count_mismatch_rows'] += 1
            count = 0
            for slot, group in enumerate(groups[:len(slots)]):
                point = position(group)
                if point is not None:
                    count += 1
                    team_counts[slot_teams[slot]] += 1
                    if not all(0 <= coordinate <= 1 for coordinate in point):
                        counters['player_positions_outside_unit_pitch'] += 1
            player_counts[count] += 1
            point = position(parts[2])
            if point is not None:
                ball_seen[frame] = 1
                if not all(0 <= coordinate <= 1 for coordinate in point):
                    counters['ball_positions_outside_unit_pitch'] += 1
            elif any(math.isfinite(v) for v in _numeric_coordinates(parts[2])):
                counters['partial_ball_coordinates'] += 1
    missing = expected_frames - sum(seen)
    report = {key: counters[key] for key in (
        'rows', 'malformed_rows', 'out_of_range_ids', 'nonsequential_rows',
        'duplicate_frame_ids', 'slot_count_mismatch_rows',
        'player_positions_outside_unit_pitch', 'ball_positions_outside_unit_pitch',
        'partial_ball_coordinates')}
    report.update({
        'expected_frames': expected_frames, 'missing_frame_ids': missing,
        'frame_alignment_ok': missing == 0 and counters['rows'] == expected_frames
            and not any(counters[k] for k in ('malformed_rows', 'out_of_range_ids',
                                            'duplicate_frame_ids', 'nonsequential_rows')),
        'track_slots': len(slots), 'ball_observed_frames': sum(ball_seen),
        'ball_observed_fraction': sum(ball_seen) / expected_frames,
        'ball_missing_runs': missing_runs(ball_seen, fps),
        'player_count_histogram': dict(sorted(player_counts.items())),
        'player_observations_by_team': team_counts,
        'team_order': 'Metadata Team order; no player identity is inferred.',
        'missing_ball_note': 'Includes absent rows and nonfinite coordinates; never a dead-ball label.',
    })
    return report


def _numeric_coordinates(group):
    values = []
    for field in group.split(',')[:2]:
        try:
            values.append(float(field))
        except ValueError:
            values.append(float('nan'))
    return values


def coverage(intervals, n):
    delta = array('i', [0]) * (n + 1)
    for start, end in intervals:
        delta[start] += 1
        delta[end] -= 1
    active = 0
    mask = bytearray(n)
    for i in range(n):
        active += delta[i]
        mask[i] = active > 0
    return mask


def audit_tags(path, expected_frames, fps):
    by_code, by_team = defaultdict(list), defaultdict(list)
    counts, teams = Counter(), Counter()
    intervals = []
    invalid = clipped = unknown = total = 0
    for item in ET.parse(path).iter('instance'):
        total += 1
        try:
            start_s, end_s = float(item.findtext('start')), float(item.findtext('end'))
            if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
                raise ValueError()
            start, end = round(start_s * fps), round(end_s * fps)
            if end <= start:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        a, b = max(0, min(start, expected_frames)), max(0, min(end, expected_frames))
        clipped += (a, b) != (start, end)
        code = (item.findtext('code') or 'UNSPECIFIED').strip()
        tags = {(label.findtext('group') or '').strip().casefold():
                (label.findtext('text') or '').strip() for label in item.findall('label')}
        team = tags.get('team', '')
        counts[code] += 1
        teams[team or 'UNSPECIFIED'] += 1
        if team.casefold() in ('', 'n/a', 'unknown'):
            unknown += 1
        elif a < b:
            by_team[team.casefold()].append((a, b))
        if a < b:
            intervals.append((a, b))
            by_code[code].append((a, b))
    team_activity = array('i', [0]) * expected_frames
    team_coverage = {}
    for team, spans in by_team.items():
        mask = coverage(spans, expected_frames)
        team_coverage[team] = sum(mask)
        for i, active in enumerate(mask):
            team_activity[i] += active
    return {
        'instances': total, 'invalid_intervals': invalid, 'clipped_intervals': clipped,
        'instances_without_known_team': unknown,
        'instances_by_code': dict(sorted(counts.items())),
        'instances_by_team_tag': dict(sorted(teams.items())),
        'covered_frames': sum(coverage(intervals, expected_frames)),
        'coverage_by_code_frames': {code: sum(coverage(spans, expected_frames))
                                    for code, spans in sorted(by_code.items())},
        'coverage_by_team_tag_frames': dict(sorted(team_coverage.items())),
        'multiple_team_tag_frames': sum(value > 1 for value in team_activity),
        'interpretation': 'Coverage is not possession. Attacking and defending tags can '
                          'legitimately overlap for opposing teams; no contradiction is inferred.',
        'interval_convention': 'round(seconds * fps), half-open [start, end), clipped to recording',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('MTAH_DATA_DIR'))
    parser.add_argument('--out', type=Path, required=True, help='Local report path; do not upload')
    parser.add_argument('--source', choices=('tactical', 'broadcast'), default='tactical')
    parser.add_argument('--expected-frames', type=int)
    parser.add_argument('--fps', type=float, default=25)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error('--data-dir or MTAH_DATA_DIR is required')
    n = args.expected_frames if args.expected_frames is not None else (
        199319 if args.source == 'tactical' else 164488)
    if n <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('Frame count and fps must be positive and finite')
    folder = args.data_dir / args.source
    stem = 'BAR-ATH_' + args.source
    try:
        tracking = audit_tracking(folder / (stem + '_atd.txt'),
                                  folder / (stem + '_atd.xml'), n, args.fps)
        tag_path = folder / (stem + '_smart_tagging.xml')
        tags = audit_tags(tag_path, n, args.fps) if tag_path.exists() else None
        report = {'source': args.source, 'scope': 'Entire recording, not live-play frames',
                  'tracking': tracking, 'smart_tagging': tags,
                  'restrictions': 'Local only; no ground truth, video, predictions or parameter tuning.'}
        inputs = {p.resolve() for p in folder.iterdir() if p.is_file()}
        if args.out.resolve() in inputs:
            raise ValueError('Report must not overwrite an input asset')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, ET.ParseError):
        print('Audit failed. Check local input paths, XML structure and output permissions. '
              'Input contents were not printed.', file=sys.stderr)
        return 1
    print('Audit report saved locally. No dataset content printed.')
    return 0 if tracking['frame_alignment_ok'] and not tracking['slot_count_mismatch_rows'] else 2


if __name__ == '__main__':
    sys.exit(main())
