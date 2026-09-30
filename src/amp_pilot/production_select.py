"""Production selector: regional quotas, explicit feature guards, monotone MIC swaps."""
import numpy as np
from .selector import CoverageSelector,allocate,ranks


def potency_utility(values,known_similarity):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('Invalid activity table')
    if np.any(values[:,:-1]<0) or np.any(values[:,:-1]>1) or np.any(values[:,-1]<=0):raise ValueError('Invalid prediction units')
    # Fixed candidate-union ranks. Raw MIC values are additionally guarded below.
    return .55*ranks(-np.log2(values[:,-1]))+.45*ranks(values[:,0])-.05*ranks(known_similarity)


class GuardedSelector:
    def __init__(self,emb,reference,props,reference_props,seed=42):
        self.cover=CoverageSelector(emb,reference,props,reference_props,seed)
        self.seed=seed;self.props=np.asarray(props);self.reference_props=np.asarray(reference_props)
        # RFF embedding discrepancy and descriptor-wise empirical CDF discrepancies.
        bank=[self.cover.phi[:,:128]];targets=[self.cover.target[:128]]
        self.groups=[('embedding_rff',slice(0,128))]
        offset=128
        for j,name in enumerate(['length','charge','hydrophobicity','moment']):
            thresholds=np.quantile(reference_props[:,j],np.linspace(.005,.995,33))
            f=(props[:,j,None]<=thresholds[None,:]).astype(np.float32)
            t=(reference_props[:,j,None]<=thresholds[None,:]).mean(0)
            bank.append(f);targets.append(t);self.groups.append((name,slice(offset,offset+len(t))));offset+=len(t)
        self.phi=np.column_stack(bank).astype(np.float32);self.target=np.concatenate(targets)

    def discrepancy(self,mean):
        d=mean-self.target
        return np.array([np.mean(d[sl]**2) if name=='embedding_rff' else np.mean(np.abs(d[sl])) for name,sl in self.groups])

    def start(self,n):
        q=allocate(n,self.cover.weights,self.cover.capacity);rng=np.random.default_rng(self.seed)
        ids=np.concatenate([rng.choice(np.flatnonzero(self.cover.bins==j),int(k),replace=False) for j,k in enumerate(q) if k])
        return ids.astype(np.int64),q

    def optimize(self,seqs,values,known_similarity,n,rounds=2000,tolerance=.10,proposals=256,accept_per_round=32):
        if n>len(seqs) or len(set(seqs))!=len(seqs):raise ValueError('Insufficient or duplicated candidates')
        baseline,quotas=self.start(n);ids=baseline.copy();mask=np.zeros(len(seqs),bool);mask[ids]=True
        utility=potency_utility(values,known_similarity);mean=self.phi[ids].mean(0,dtype=np.float64)
        base_disc=self.discrepancy(mean);caps=np.maximum(base_disc*(1+tolerance),1e-6)
        amp_floor=values[ids,:-1].mean(0)-.01;novelty_cap=float(np.mean(known_similarity[ids]))
        amp_mean=values[ids,:-1].mean(0);sim_mean=novelty_cap
        slots=[np.flatnonzero(self.cover.bins[ids]==j) for j in range(len(quotas))]
        rng=np.random.default_rng(self.seed+710);accepted=0;history=[];snapshots=[baseline.copy()]
        # Most incoming proposals are drawn from the better-potency half; keep broad exploration.
        good=np.argsort(-utility,kind='stable')[:max(n,len(seqs)//2)]
        for step in range(rounds):
            incoming=np.concatenate([rng.choice(good,proposals//2),rng.integers(len(seqs),size=proposals-proposals//2)])
            incoming=incoming[~mask[incoming]]
            candidates=[]
            for i in incoming:
                available=slots[self.cover.bins[i]]
                if len(available):
                    slot=int(rng.choice(available));o=ids[slot]
                    if values[i,-1]<=values[o,-1] and utility[i]>utility[o]:candidates.append((float(utility[i]-utility[o]),int(i),slot))
            candidates.sort(reverse=True);used=set();taken=0
            for gain,i,slot in candidates:
                if taken>=accept_per_round:break
                if mask[i] or slot in used:continue
                o=ids[slot]
                if values[i,-1]>values[o,-1] or utility[i]<=utility[o]:continue
                proposed=mean+(self.phi[i].astype(float)-self.phi[o])/n
                amp_new=amp_mean+(values[i,:-1]-values[o,:-1])/n
                sim_new=sim_mean+(known_similarity[i]-known_similarity[o])/n
                if np.any(self.discrepancy(proposed)>caps+1e-12) or np.any(amp_new<amp_floor) or sim_new>novelty_cap+1e-12:continue
                mask[o]=False;mask[i]=True;ids[slot]=i;mean=proposed;amp_mean=amp_new;sim_mean=sim_new
                accepted+=1;taken+=1;used.add(slot)
            if (step+1)%250==0 or step+1==rounds:
                record={'round':step+1,'accepted':accepted,'mic_mean':float(values[ids,-1].mean()),'mic_median':float(np.median(values[ids,-1])),'mic_le16':float(np.mean(values[ids,-1]<=16)),'feature_discrepancies':dict(zip([k for k,_ in self.groups],map(float,self.discrepancy(mean))))}
                history.append(record);snapshots.append(ids.copy());print('Selection',record,flush=True)
        return baseline,ids,{'quotas':quotas.tolist(),'reference_region_weights':self.cover.weights.tolist(),'region_counts':self.cover.capacity.tolist(),'quota_l1':float(np.abs(quotas/n-self.cover.weights).sum()),'feature_caps':dict(zip([k for k,_ in self.groups],map(float,caps))),'amp_probability_floors':amp_floor.tolist(),'known_similarity_mean_cap':novelty_cap,'accepted_swaps':accepted,'trace':history},snapshots


def max_similarity(seqs,reference,workers=4,block=128):
    from rapidfuzz import process,fuzz
    if not reference:raise ValueError('Empty similarity reference')
    out=np.empty(len(seqs),np.float32)
    for start in range(0,len(seqs),block):
        m=process.cdist(seqs[start:start+block],reference,scorer=fuzz.ratio,workers=workers,dtype=np.float32)
        out[start:start+len(m)]=m.max(1)/100
        if start%(block*100)==0:print('Known similarity',start,'/',len(seqs),flush=True)
    return out


def neighbor_stats(seqs,workers=4,block=128,return_counts=False):
    """Exhaustive selected-set normalized Indel neighbors, without an NxN allocation."""
    from rapidfuzz import process,fuzz
    n=len(seqs);maximum=np.zeros(n);counts=np.zeros(n,np.int64)
    if n<1:raise ValueError("Empty selected set")
    for start in range(0,n,block):
        matrix=process.cdist(seqs[start:start+block],seqs,scorer=fuzz.ratio,workers=workers,dtype=np.float32)
        matrix[np.arange(len(matrix)),np.arange(start,start+len(matrix))]=-1
        maximum[start:start+len(matrix)]=np.maximum(matrix.max(1),0)/100
        counts[start:start+len(matrix)]=(matrix>80).sum(1)
        if start%(block*100)==0:print('Full selected neighbors',start,'/',n,flush=True)
    report={'n':n,'near_neighbor_fraction_gt_0_8':float(np.mean(counts>0)),'mean_neighbors_gt_0_8':float(counts.mean()),'max_similarity_median':float(np.median(maximum)),'nearest_distance_median':float(np.median(1-maximum)),'nearest_distance_q10':float(np.quantile(1-maximum,.1)),'undirected_edges_gt_0_8':int(counts.sum()//2),'definition':'Exhaustive all selected pairs, normalized Indel similarity; >0.8 threshold, not official alignment identity.'}
    return (report,counts) if return_counts else report


def repair_neighbors(selector,seqs,ids,baseline,values,known,report,workers=4,budget=3000):
    """Repair actual selected edges within region, checking every proposed incoming sequence."""
    from rapidfuzz import process,fuzz
    ids=ids.copy();n=len(ids);chosen=[seqs[i] for i in ids]
    before,counts=neighbor_stats(chosen,workers,return_counts=True)
    utility=potency_utility(values,known);mask=np.zeros(len(seqs),bool);mask[ids]=True
    queues=[np.array(sorted(np.flatnonzero(selector.cover.bins==j),key=lambda i:(-utility[i],seqs[i])),dtype=np.int64) for j in range(len(selector.cover.weights))]
    positions=np.zeros(len(queues),int);mean=selector.phi[ids].mean(0,dtype=np.float64)
    caps=np.array([report['feature_caps'][k] for k,_ in selector.groups]);amp_floor=np.array(report['amp_probability_floors']);sim_cap=report['known_similarity_mean_cap']
    amp=values[ids,:-1].mean(0);sim=float(known[ids].mean());total_mic=float(values[ids,-1].sum());mic_cap=min(float(values[baseline,-1].sum()),total_mic*1.05)
    fixed=0;attempts=0
    while attempts<budget and counts.max(initial=0)>0:
        # Remove the worst connected member; prefer replacing the weaker prediction on ties.
        active=(positions<np.array([len(q) for q in queues]))[selector.cover.bins[ids]] & (counts>0)
        eligible=np.flatnonzero(active)
        if not len(eligible):break
        tied=eligible[counts[eligible]==counts[eligible].max()]
        slot=int(tied[np.argmax(values[ids[tied],-1])]);o=ids[slot];region=selector.cover.bins[o]
        queue=queues[region];accepted=False
        while positions[region]<len(queue) and attempts<budget:
            i=int(queue[positions[region]]);positions[region]+=1
            if mask[i]:continue
            attempts+=1
            proposed=mean+(selector.phi[i].astype(float)-selector.phi[o])/n
            amp_new=amp+(values[i,:-1]-values[o,:-1])/n;sim_new=sim+(known[i]-known[o])/n
            if np.any(selector.discrepancy(proposed)>caps+1e-12) or np.any(amp_new<amp_floor) or sim_new>sim_cap+1e-12 or total_mic+values[i,-1]-values[o,-1]>mic_cap:continue
            neighbors=process.cdist([seqs[i]],chosen,scorer=fuzz.ratio,workers=workers,dtype=np.float32)[0]>80;neighbors[slot]=False
            if neighbors.sum()>=counts[slot]:continue
            old=process.cdist([seqs[o]],chosen,scorer=fuzz.ratio,workers=workers,dtype=np.float32)[0]>80;old[slot]=False
            counts[old]-=1;counts[neighbors]+=1;counts[slot]=int(neighbors.sum())
            mask[o]=False;mask[i]=True;ids[slot]=i;chosen[slot]=seqs[i]
            mean=proposed;amp=amp_new;sim=sim_new;total_mic+=values[i,-1]-values[o,-1];fixed+=1;accepted=True
            if fixed%100==0:print('Neighbor repair',fixed,'accepted;',attempts,'proposals',flush=True)
            break
    after=neighbor_stats(chosen,workers)
    return ids,{'before':before,'after':after,'accepted':fixed,'attempted':attempts,'budget':budget,'warning':'Finite-budget repair; remaining close neighbors are reported, not silently declared absent.'}
