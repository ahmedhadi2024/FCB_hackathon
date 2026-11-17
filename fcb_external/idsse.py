"""Two-pass streaming DFL adapter. No ground-truth dependency in observation construction."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .schema import Observations, Targets

PERIODS = {'firstHalf':1, 'secondHalf':2}


def localname(tag):
    return tag.rsplit('}',1)[-1]


def metadata(path, match_id):
    root = ET.parse(path).getroot()
    general = next(root.iter('General'), None)
    env = next(root.iter('Environment'), None)
    if general is None or env is None:
        raise ValueError('Missing DFL match metadata')
    expected = 'DFL-MAT-'+match_id
    if general.get('MatchId') != expected:
        raise ValueError('Wrong match in metadata')
    home, away = general.get('HomeTeamId'), general.get('GuestTeamId',general.get('AwayTeamId'))
    if not home or not away or home == away:
        raise ValueError('Missing or ambiguous home/away metadata')
    players = {}
    for team in root.iter('Team'):
        tid = team.get('TeamId')
        if tid not in (home,away):
            continue
        for player in team.findall('Players/Player'):
            pid = player.get('PersonId')
            if not pid or pid in players:
                raise ValueError('Duplicate/missing player ID in metadata')
            players[pid] = (0 if tid==home else 1, tid)
    if not players or {v[0] for v in players.values()} != {0,1}:
        raise ValueError('Metadata must supply both team rosters')
    ids = sorted(players,key=lambda pid:(players[pid][0],pid))
    return {'home':home,'away':away,'players':players,'ids':ids,
            'pitch':(float(env.get('PitchX')),float(env.get('PitchY')))}


def records(path, match_id):
    """Yield a frame and its FrameSet identity without retaining XML children."""
    stack = []
    context = None
    with Path(path).open('rb') as handle:
        for event, elem in ET.iterparse(handle,events=('start','end')):
            tag = localname(elem.tag)
            if event == 'start':
                stack.append(elem)
                if elem.get('MatchId') and elem.get('MatchId') != 'DFL-MAT-'+match_id:
                    raise ValueError('Wrong match in position XML')
                if tag == 'FrameSet':
                    context = dict(elem.attrib)
                continue
            if tag == 'Frame':
                if context is None:
                    raise ValueError('Frame outside a FrameSet')
                yield context, dict(elem.attrib)
            if tag == 'FrameSet':
                context = None
            stack.pop()
            if tag in ('Frame','FrameSet'):
                if stack:
                    stack[-1].remove(elem)
                elem.clear()


def timestamp_us(value):
    try:
        dt = datetime.fromisoformat(value.replace('Z','+00:00'))
        if dt.tzinfo is None:
            raise ValueError('timestamp must include UTC offset')
        epoch = datetime(1970,1,1,tzinfo=timezone.utc)
        delta = dt.astimezone(timezone.utc)-epoch
        return (delta.days*86400+delta.seconds)*1000000+delta.microseconds
    except (ValueError,AttributeError) as exc:
        raise ValueError('Invalid tracking timestamp') from exc


def point(frame):
    try:
        pair = (float(frame.get('X')),float(frame.get('Y')))
        return pair if np.isfinite(pair).all() else (np.nan,np.nan)
    except (TypeError,ValueError):
        return np.nan,np.nan


def target_label(status, possession):
    # Unknown values remain X. No interpolation or label-derived player selection.
    if status == '0':
        return 'D'
    if status == '1':
        return {'1':'H','2':'A'}.get(possession,'X')
    return 'X'


def entity(context, meta):
    tid = context.get('TeamId','')
    if tid.casefold() == 'ball':
        return 'ball'
    pid = context.get('PersonId')
    if tid not in (meta['home'],meta['away']):
        return None  # Explicit non-team objects (e.g. referees) are not players.
    if pid not in meta['players'] or meta['players'][pid][1] != tid:
        raise ValueError('Unknown player or conflicting team membership in FrameSet')
    return pid


def convert(position_path, metadata_path, match_id):
    meta = metadata(metadata_path,match_id)
    timeline = {}
    ignored = 0
    for context, frame in records(position_path,match_id):
        obj = entity(context,meta)
        if obj is None:
            ignored += 1
            continue
        if context.get('GameSection') not in PERIODS:
            raise ValueError('Unsupported period; expected firstHalf or secondHalf')
        period = PERIODS[context['GameSection']]
        key = (period,int(frame['N']))
        stamp = timestamp_us(frame.get('T'))
        if key in timeline and timeline[key] != stamp:
            raise ValueError('Conflicting timestamp for a shared frame key')
        timeline[key] = stamp
    if not timeline:
        raise ValueError('No tracking observations')
    keys = sorted(timeline)
    index = {key:i for i,key in enumerate(keys)}
    n, p = len(keys),len(meta['ids'])
    period = np.array([key[0] for key in keys],dtype=np.int8)
    frame_id = np.array([key[1] for key in keys],dtype=np.int64)
    time_s = np.empty(n,dtype=np.float64)
    sequence = np.zeros(n,dtype=np.int32)
    starts = {}
    for i,key in enumerate(keys):
        starts.setdefault(key[0],timeline[key])
        time_s[i] = (timeline[key]-starts[key[0]])/1e6
        if i:
            same = period[i] == period[i-1]
            if same:
                dn = int(frame_id[i]-frame_id[i-1])
                dt = timeline[key]-timeline[keys[i-1]]
                if dn<=0 or abs(dt-dn*40000)>1:
                    raise ValueError('Timestamps/frame IDs are inconsistent with 25 Hz')
            sequence[i] = sequence[i-1]+int(not same or frame_id[i]!=frame_id[i-1]+1)
    player_xy = np.full((n,p,2),np.nan,dtype=np.float32)
    ball_xy = np.full((n,2),np.nan,dtype=np.float32)
    used = np.zeros((n,p+1),dtype=bool)
    labels = np.full(n,'X',dtype='<U1')
    slots = {pid:i for i,pid in enumerate(meta['ids'])}
    statuses, possessions = Counter(),Counter()
    ball_rows = 0
    for context, frame in records(position_path,match_id):
        obj = entity(context,meta)
        if obj is None:
            continue
        i = index[(PERIODS[context['GameSection']],int(frame['N']))]
        slot = p if obj=='ball' else slots[obj]
        if used[i,slot]:
            raise ValueError('Duplicate frame/entity record; refusing silent aggregation')
        used[i,slot] = True
        if obj == 'ball':
            ball_xy[i] = point(frame)
            status = frame.get('BallStatus','MISSING')
            possession = frame.get('BallPossession','MISSING')
            statuses[status] += 1
            possessions[possession] += 1
            labels[i] = target_label(status,possession)
            ball_rows += 1
        else:
            player_xy[i,slot] = point(frame)
    if n < 2 or not any(period[i]==period[i-1] for i in range(1,n)):
        raise ValueError('Insufficient timestamps to verify 25 Hz')
    obs = Observations(match_id,period,frame_id,time_s,player_xy,
                       np.isfinite(player_xy).all(axis=-1),
                       np.array([meta['players'][pid][0] for pid in meta['ids']],dtype=np.int8),
                       ball_xy,np.isfinite(ball_xy).all(axis=-1),sequence,
                       *meta['pitch'],25.0).validate()
    targets = Targets(labels,labels!='X').validate(n)
    audit = {'status_domains':dict(statuses),'possession_domains':dict(possessions),
             'ball_rows':ball_rows,'frames':n,'player_slots':p,
             'ignored_nonteam_records':ignored,'sequences':int(sequence[-1])+1,
             'valid_target_frames':int(targets.valid.sum()),
             'label_counts':dict(Counter(labels.tolist())),
             'coordinate_convention':'DFL centred pitch coordinates, metres, no direction flip',
             'time_convention':'Seconds from first observed frame in each period; original N retained',
             'target_provenance':'External DFL fields; human/algorithmic ownership annotation provenance not established',
             'note':'Missing ball observations never determine target validity or sequence splitting.'}
    return obs,targets,audit
