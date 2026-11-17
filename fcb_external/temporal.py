"""Small CPU temporal convolutional network and sequence-local decoding."""
import os
import numpy as np
import torch
from torch import nn
from .features import sequence_spans


# Set FCB_DEVICE=cuda to train and score on a GPU. The default (cpu) is the environment that
# generated the submitted file; GPU results are not bit-identical to CPU results.
DEVICE=os.environ.get('FCB_DEVICE','cpu')


def configure(seed,threads):
    if DEVICE!='cpu':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=DEVICE!='cpu')


class TemporalNet(nn.Module):
    def __init__(self,features,channels,dilations):
        super().__init__()
        self.radius=sum(dilations)
        self.input=nn.Conv1d(features,channels,1)
        self.blocks=nn.ModuleList([nn.Conv1d(channels,channels,3,padding=d,dilation=d)
                                   for d in dilations])
        self.output=nn.Conv1d(channels,3,1)

    def forward(self,x,observation_mask=None):
        x=torch.relu(self.input(x))
        if observation_mask is not None:
            x=x*observation_mask
        for block in self.blocks:
            x=torch.relu(x+block(x))
            if observation_mask is not None:
                x=x*observation_mask
        return self.output(x)


def probabilities(net,x,sequence_id,chunk_frames=4096):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or len(x)!=len(sequence_id) or not np.isfinite(x).all() or chunk_frames<1:
        raise ValueError('Finite aligned features and positive chunk size required')
    net.eval(); net.to(DEVICE)
    result=np.empty((len(x),3),np.float32)
    with torch.inference_mode():
        for lo,hi in sequence_spans(sequence_id):
            for start in range(lo,hi,chunk_frames):
                end=min(hi,start+chunk_frames)
                left=max(lo,start-net.radius); right=min(hi,end+net.radius)
                block=torch.from_numpy(np.ascontiguousarray(x[left:right].T))[None].to(DEVICE)
                pred=torch.softmax(net(block),dim=1)[0].T.cpu().numpy()
                result[start:end]=pred[start-left:end-left]
    return result


def sample_windows(x,y,seq,rng,batch_size,core_frames,radius,return_mask=False):
    """Random cores, same-sequence halo, zero feature padding and ignored halo targets."""
    edges=np.r_[0,np.flatnonzero(np.diff(seq))+1,len(seq)]
    length=core_frames+2*radius
    windows=np.zeros((batch_size,x.shape[1],length),np.float32)
    labels=np.full((batch_size,length),-100,np.int64)
    mask=np.zeros((batch_size,1,length),np.float32)
    for i in range(batch_size):
        anchor=int(rng.integers(len(seq)))
        s=int(np.searchsorted(edges,anchor,side='right')-1)
        lo,hi=int(edges[s]),int(edges[s+1])
        core=anchor-core_frames//2; start=core-radius
        left=max(lo,start); right=min(hi,start+length)
        windows[i,:,left-start:right-start]=x[left:right].T
        mask[i,:,left-start:right-start]=1
        left=max(lo,core); right=min(hi,core+core_frames)
        labels[i,left-start:right-start]=y[left:right]
    return (windows,labels,mask) if return_mask else (windows,labels)


def decode(probabilities,sequence_id,penalty):
    """Viterbi with a fixed cost for changing class, reset at every sequence."""
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(len(sequence_id),3) or not np.isfinite(p).all() or np.any(p<0) or \
            not np.allclose(p.sum(axis=1),1,atol=1e-5) or not np.isfinite(penalty) or penalty<0:
        raise ValueError('Invalid probabilities or transition penalty')
    if penalty==0:
        return np.array(['H','A','D'])[p.argmax(axis=1)]
    result=np.empty(len(p),np.int8)
    transition=np.full((3,3),-float(penalty)); np.fill_diagonal(transition,0)
    for lo,hi in sequence_spans(sequence_id):
        emission=np.log(np.maximum(p[lo:hi],1e-8))
        back=np.empty((hi-lo,3),np.int8); score=emission[0]
        for i in range(1,hi-lo):
            options=score[:,None]+transition
            back[i]=options.argmax(axis=0)
            score=options.max(axis=0)+emission[i]
        state=int(score.argmax())
        for i in range(hi-lo-1,-1,-1):
            result[lo+i]=state
            if i:
                state=int(back[i,state])
    return np.array(['H','A','D'])[result]
