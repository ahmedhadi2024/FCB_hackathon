"""Controlled-possession convention from the organisers' published CHALLENGE.md.

The published rule: possession changes only when the other team gains *controlled*
possession; a simple touch, a deflection or an aerial duel won without control is not a
change, and while the ball is loose or contested possession stays with the last team.
After a stoppage there is a clear possessor.

absorb_short_spells applies that rule mechanically to any H/A/D/X label sequence:
an opposing live spell shorter than min_frames is given back to the team already in
possession. It is used on external training targets (whose provider flips ownership on
single touches) and on decoded predictions. It reads no competition answers.
"""
import numpy as np

LIVE=('H','A')


def runs(labels,sequence_id):
    """(start, stop) of maximal constant-label runs that never cross a sequence."""
    labels=np.asarray(labels); seq=np.asarray(sequence_id)
    change=(labels[1:]!=labels[:-1])|(seq[1:]!=seq[:-1])
    edges=np.r_[0,np.flatnonzero(change)+1,len(labels)]
    return list(zip(edges[:-1].tolist(),edges[1:].tolist()))


def _absorb_once(labels,sequence_id,min_frames):
    out=labels.copy(); owner=None; previous_sequence=None
    for start,stop in runs(labels,sequence_id):
        label=labels[start]; sequence=sequence_id[start]
        if sequence!=previous_sequence:
            owner=None; previous_sequence=sequence
        if label not in LIVE:
            owner=None          # dead ball or unknown: the next live spell is a restart
        elif owner is None or label==owner:
            owner=label         # restart possessor, or the same team continuing
        elif stop-start<min_frames:
            out[start:stop]=owner   # uncontrolled touch: possession stays with the last team
        else:
            owner=label         # controlled change of possession
    return out


def absorb_short_spells(labels,sequence_id,min_frames):
    """Return labels with uncontrolled opposing spells (< min_frames) absorbed.

    Two passes: tiny blips (< min_frames // 5) are removed first, so a real spell that the
    labels interrupt for a frame or two is measured at its full length in the second pass.
    D and X frames are never changed, and the first live spell after a stoppage is kept.
    """
    labels=np.asarray(labels); sequence_id=np.asarray(sequence_id)
    if labels.ndim!=1 or labels.shape!=sequence_id.shape or not len(labels):
        raise ValueError('Aligned one-dimensional labels and sequence IDs required')
    if not isinstance(min_frames,(int,np.integer)) or isinstance(min_frames,bool) or min_frames<0:
        raise ValueError('Nonnegative integer minimum controlled duration required')
    if not np.isin(labels,['H','A','D','X']).all():
        raise ValueError('Unsupported possession label')
    if min_frames<=1:
        return labels.copy()
    out=labels
    for threshold in sorted({max(2,min_frames//5),min_frames}):
        out=_absorb_once(out,sequence_id,threshold)
    return out


def shift_dead_edges(labels,sequence_id,start_earlier,end_later):
    """Move every dead-ball run's start earlier and its end later by fixed frame counts.

    The training provider and the organiser mark the moment the ball goes dead and comes back
    slightly differently; the offsets are measured on the organiser's own public sample matches
    (other games). A run never grows across a sequence boundary or into another dead run, and
    X frames are never changed. Negative values are not supported.
    """
    labels=np.asarray(labels); sequence_id=np.asarray(sequence_id)
    for value in (start_earlier,end_later):
        if not isinstance(value,(int,np.integer)) or isinstance(value,bool) or value<0:
            raise ValueError('Nonnegative integer edge shifts required')
    if labels.shape!=sequence_id.shape or labels.ndim!=1:
        raise ValueError('Aligned one-dimensional labels and sequence IDs required')
    out=labels.copy()
    if not (start_earlier or end_later):
        return out
    dead=[(a,b) for a,b in runs(labels,sequence_id) if labels[a]=='D']
    for a,b in dead:
        lo=a
        while lo>0 and a-lo<start_earlier and sequence_id[lo-1]==sequence_id[a] and labels[lo-1] in LIVE:
            lo-=1
        hi=b
        while hi<len(labels) and hi-b<end_later and sequence_id[hi]==sequence_id[b-1] and labels[hi] in LIVE:
            hi+=1
        out[lo:a]='D'; out[b:hi]='D'
    return out
