"""Optional automatic phase evidence; no ground-truth or video access."""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
import numpy as np


def normalized(value):
    return ' '.join((value or '').split()).casefold()


@dataclass(frozen=True)
class Tag:
    code: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    team: str | None
    half: str | None
    kind: str | None
    conflicts: tuple[str,...]


def load(path,fps=25):
    if fps!=25:raise ValueError('Expected tactical 25 fps clock')
    out=[]
    for item in ET.parse(path).getroot().iter('instance'):
        try:start=float(item.findtext('start'));end=float(item.findtext('end'))
        except (TypeError,ValueError):raise ValueError('Invalid tag timestamp') from None
        if not math.isfinite(start) or not math.isfinite(end) or start<0 or end<=start:raise ValueError('Invalid tag interval')
        groups={}
        for label in item.findall('label'):
            key=normalized(label.findtext('group'));value=normalized(label.findtext('text'))
            groups.setdefault(key,set()).add(value)
        conflicts=tuple(sorted(k for k,v in groups.items() if len(v)>1))
        def field(key):
            values=groups.get(key,set())
            return next(iter(values)) if len(values)==1 and not values&{'','n/a','unknown'} else None
        out.append(Tag(normalized(item.findtext('code')),start,end,round(start*fps),round(end*fps),field('team'),field('half'),field('type'),conflicts))
    return out


def metadata_team_map(path):
    teams=list(ET.parse(path).getroot().iter('Team'))
    if len(teams)!=2:raise ValueError('Expected two teams in documented home-away order')
    names=[normalized(t.get('name') or t.findtext('Name')) for t in teams]
    ids=[t.get('id') for t in teams]
    if not all(names) or names[0]==names[1] or not all(ids) or ids[0]==ids[1]:raise ValueError('Explicit distinct metadata names and IDs required')
    return {names[0]:0,names[1]:1}


def owner_evidence(tags,frames,period,team_map,margin_frames=25):
    frames=np.asarray(frames);period=np.asarray(period)
    if frames.ndim!=1 or period.shape!=frames.shape or margin_frames<0:raise ValueError('Invalid evidence timeline')
    mapping={normalized(k):v for k,v in team_map.items()}
    if any(v not in (0,1) for v in mapping.values()):raise ValueError('Invalid team side')
    present=np.zeros((len(frames),2),bool)
    for tag in tags:
        # Only two documented attacking phase types; no defender-side inversion.
        if tag.code not in ('build up','progression') or tag.conflicts:continue
        side=mapping.get(tag.team);half={'1st half':1,'2nd half':2}.get(tag.half)
        if side is None or half is None:continue
        use=(frames>=tag.start_frame+margin_frames)&(frames<tag.end_frame-margin_frames)&(period==half)
        present[use,side]=True
    result=np.full(len(frames),-1,np.int8)
    result[present[:,0]&~present[:,1]]=0;result[present[:,1]&~present[:,0]]=1
    return result


def kickoff_dead_mask(tags,frames,period,max_lead_frames=15000):
    """Frames of a half that precede its tagged 'Kick Off (Start)' window: the ball is dead.

    The automatic tag marks a window around the restart, so only frames before the window
    opens are marked. A half is left untouched unless it has exactly one such tag, the tag
    opens inside that half, and it opens within max_lead_frames of the half's first frame.
    No frame number of any particular match is used.
    """
    frames=np.asarray(frames);period=np.asarray(period)
    if frames.ndim!=1 or period.shape!=frames.shape or max_lead_frames<0:raise ValueError('Invalid kickoff timeline')
    mask=np.zeros(len(frames),bool)
    for name,half in (('1st half',1),('2nd half',2)):
        inside=period==half
        found=[t for t in tags if t.code=='set pieces' and t.kind=='kick off (start)' and t.half==name and not t.conflicts]
        if len(found)!=1 or not inside.any():continue
        first=int(frames[inside].min());last=int(frames[inside].max());start=found[0].start_frame
        if first<start<=min(last,first+max_lead_frames):
            mask|=inside&(frames<start)
    return mask


def fuse(probabilities,side,ball_missing,odds_multiplier=1.5):
    p=np.asarray(probabilities);side=np.asarray(side);missing=np.asarray(ball_missing)
    if not np.issubdtype(p.dtype,np.floating):p=p.astype(float)
    if p.ndim!=2 or p.shape[1]!=3 or side.shape!=(len(p),) or missing.shape!=side.shape:raise ValueError('Invalid evidence shapes')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('Invalid probabilities')
    if not math.isfinite(odds_multiplier) or not 1<=odds_multiplier<=2:raise ValueError('Bounded odds multiplier required')
    if not np.isin(side,[-1,0,1]).all() or missing.dtype!=np.bool_:raise ValueError('Invalid side/missing mask')
    result=p.copy()
    for team in (0,1):
        rows=np.flatnonzero(missing&(side==team)&(p[:,:2].sum(1)>0))
        live=p[rows,:2].copy();mass=live.sum(1);live[:,team]*=odds_multiplier
        result[rows,:2]=live/live.sum(1)[:,None]*mass[:,None]
    return result


def apply_candidate(probabilities,obs,tags_path,metadata_path,config):
    """Apply the single frozen prototype, or preserve the disabled baseline exactly."""
    if config is None:return probabilities
    expected={'phase_codes':['build up','progression'],'margin_frames':25,'odds_multiplier':1.5,
              'missing_ball_only':True,'preserve_dead_probability':True,'conflict_policy':'unchanged_v2'}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported candidate policy')
    tags=load(tags_path);mapping=metadata_team_map(metadata_path)
    side=owner_evidence(tags,obs.frame_id-1,obs.period,mapping,config['margin_frames'])
    return fuse(probabilities,side,~obs.ball_observed,config['odds_multiplier'])
