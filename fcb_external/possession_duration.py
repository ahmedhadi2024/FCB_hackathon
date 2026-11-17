"""Experimental soft possession-duration decoding; never fits public labels."""
import numpy as np
from .temporal import decode as baseline
from .features import sequence_spans


def decode(probabilities,sequence_id,transition_penalty=10,min_frames=12,short_penalty=2):
    p=np.asarray(probabilities,dtype=float);seq=np.asarray(sequence_id)
    if not isinstance(min_frames,int) or not 1<=min_frames<=100 or not np.isfinite(short_penalty) or short_penalty<0:
        raise ValueError('Bounded integer duration and nonnegative finite penalty required')
    if seq.ndim!=1 or not len(seq) or not np.issubdtype(seq.dtype,np.integer) or seq[0]!=0 or not np.isin(np.diff(seq),[0,1]).all():
        raise ValueError('Consecutive sequence IDs required')
    # Baseline validates probabilities and transition cost, and gives exact disabled parity.
    original=baseline(p,seq,transition_penalty)
    if short_penalty==0 or min_frames==1:return original
    k=min_frames;states=3*k;classes=np.repeat(np.arange(3),k)
    transitions=np.full((states,states),-np.inf)
    for state in range(states):
        team,age=divmod(state,k)
        transitions[state,team*k+min(age+1,k-1)]=0
        for other in range(3):
            if other!=team:
                extra=short_penalty if team!=2 and age<k-1 else 0
                transitions[state,other*k]=-transition_penalty-extra
    result=np.empty(len(p),dtype='<U1');names=np.array(['H','A','D'])
    for lo,hi in sequence_spans(seq):
        emission=np.log(np.maximum(p[lo:hi],1e-8))
        score=np.full(states,-np.inf)
        # The first spell is left-censored: its unknown pre-sequence age is mature.
        score[np.arange(3)*k+k-1]=emission[0]
        back=np.zeros((hi-lo,states),np.int16)
        for i in range(1,hi-lo):
            options=score[:,None]+transitions
            back[i]=options.argmax(0)
            score=options.max(0)+emission[i,classes]
        state=int(score.argmax())
        # No exit cost for the final right-censored spell.
        for i in range(hi-lo-1,-1,-1):
            result[lo+i]=names[classes[state]]
            if i:state=int(back[i,state])
    return result
