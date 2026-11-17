"""Session 1, step 3 diagnostics on Metrica sample game 3 (external data only).

  A  boundary ceiling: take v9e's predictions, snap every matched possession (same team, temporal
     IoU >= 0.3) to the true start and end, rescore. This is the most a perfect-timing fix could add.
  B  ball recovery simulation: restore the true ball (plus 1 m Gaussian noise) on a share of the
     frames the target_like40 scenario hid, in 2 s pieces, then rescore. Shares 0, 30, 50, 100%.

    python run_diag3.py
"""
from dataclasses import replace
import numpy as np
import torch
import run_bench as rb
import run_retrain_v8 as v8
from fcb_external import metrica_sample, retrain_v4, temporal, tracking_faults
from fcb_external.feature_recovery_experiment import seed_for
from fcb_external.schema import write_json

ROOT=rb.ROOT


def spells(labels,seq):
    out=[]; i=0; n=len(labels)
    while i<n:
        j=i
        while j+1<n and labels[j+1]==labels[i] and seq[j+1]==seq[i]:
            j+=1
        if labels[i] in ('H','A'):
            out.append((i,j+1,labels[i]))
        i=j+1
    return out


def snap(pred,truth,seq,threshold=0.3):
    new=pred.copy(); ts=spells(truth,seq); used=set()
    for lo,hi,lab in spells(pred,seq):
        best=None; bi=0
        for k,(a,b,l) in enumerate(ts):
            if l!=lab or k in used:
                continue
            inter=min(hi,b)-max(lo,a)
            if inter<=0:
                continue
            iou=inter/(max(hi,b)-min(lo,a))
            if iou>bi:
                bi=iou; best=k
        if best is not None and bi>=threshold:
            used.add(best); a,b,_=ts[best]
            start,end=min(lo,a),max(hi,b)
            new[start:end]=truth[start:end]
    return new


def restore(held,true_ball,share,seed):
    hidden=~held.ball_observed
    if share==0:
        return held
    ball=np.array(held.ball_xy_m); rng=np.random.default_rng(seed)
    piece=50                                            # 2 s at 25 Hz
    for start in range(0,len(ball),piece):
        sel=hidden[start:start+piece]
        if sel.any() and (share>=1 or rng.random()<share):
            idx=np.flatnonzero(sel)+start
            ball[idx]=true_ball[idx]+rng.normal(0,1.0,(len(idx),2))
    return replace(held,ball_xy_m=ball,ball_observed=np.isfinite(ball).all(axis=-1)).validate()


def main():
    cfg=v8.config(); temporal.configure(20261117,4)
    torch.set_num_threads(8)
    model=torch.load(ROOT/'models/temporal_v9e.pt',map_location='cpu',weights_only=True)
    obs,events=metrica_sample.load_epts_game(ROOT/'external-data/metrica_sample')
    truth=metrica_sample.event_labels(obs,events); seq=np.asarray(obs.sequence_id)
    held_obs=v8.hold_ball(obs); true_ball=np.array(held_obs.ball_xy_m)
    scenario=next(s for s in cfg['scenarios'] if s['name']=='target_like40')
    rows=[]
    for rep in range(3):
        held=v8.changed(held_obs,scenario,f'diag{rep}','metrica3')
        print('hidden share %.3f'%(1-held.ball_observed.mean()),flush=True)
        for share in (0,0.3,0.5,1.0):
            work=restore(held,true_ball,share,rep)
            work=tracking_faults.apply(work,seed_for('faults-diag','metrica3',str(rep),'target_like40'))
            pred=rb.predict(model,work,seq,{})
            s=rb.score(truth,pred,seq)['spellF1']
            snapped=rb.score(truth,snap(pred,np.asarray(truth),seq),seq)['spellF1']
            rows.append({'rep':rep,'recovered_share':share,'spellF1':s,'boundary_ceiling':snapped})
            print(f'rep {rep} recovered {share:.0%}: spellF1 {s:.4f}   with perfect boundaries {snapped:.4f}',flush=True)
    write_json(ROOT/'reports/bench/diag3.json',{'rows':rows})
    print('\nmean over 3 masks')
    for share in (0,0.3,0.5,1.0):
        r=[x for x in rows if x['recovered_share']==share]
        print(f'recovered {share:>4.0%}: spellF1 {np.mean([x["spellF1"] for x in r]):.4f}   perfect-boundary ceiling {np.mean([x["boundary_ceiling"] for x in r]):.4f}')


if __name__=='__main__':
    main()
