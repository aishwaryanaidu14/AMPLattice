"""Development-only coverage quotas and monotone random-feature swap optimization."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def ranks(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite utility')
    return (rankdata(x,method='average')-.5)/max(1,len(x))


def rff_fit(reference,seed,dimension=128):
    reference=np.asarray(reference,dtype=np.float64)
    rng=np.random.default_rng(seed)
    sub=reference[rng.choice(len(reference),min(len(reference),512),replace=False)]
    dist=cdist(sub,sub,'sqeuclidean'); positive=dist[dist>0]
    bandwidth=max(float(np.median(positive)) if len(positive) else 1.,1e-8)
    w=rng.normal(size=(reference.shape[1],dimension))/np.sqrt(bandwidth)
    b=rng.uniform(0,2*np.pi,dimension)
    return w,b


def rff(x,fit):
    w,b=fit
    return (np.sqrt(2/len(b))*np.cos(np.asarray(x)@w+b)).astype(np.float32)


def allocate(n,weights,capacity):
    """Capped proportional water filling, then true largest-remainder rounding.

    No quota can exceed capacity. Small regions receive their fractional
    remainder rather than repeatedly giving leftovers to the largest weight.
    """
    capacity=np.asarray(capacity,dtype=int); weights=np.asarray(weights,dtype=float)
    if weights.ndim!=1 or capacity.shape!=weights.shape or not np.isfinite(weights).all():
        raise ValueError('Invalid quota arrays')
    if np.any(weights<0) or np.any(capacity<0) or n>capacity.sum() or n<1:
        raise ValueError('Invalid selection count/weights/capacity')
    continuous=np.zeros(len(weights),dtype=float);active=capacity>0
    remaining=float(n)
    while remaining>1e-9 and active.any():
        w=np.where(active,weights,0.)
        if w.sum()==0:w=active.astype(float)
        target=remaining*w/w.sum()
        saturated=active & (target>=capacity-1e-12)
        if saturated.any():
            continuous[saturated]=capacity[saturated]
            remaining-=float(capacity[saturated].sum());active[saturated]=False
        else:
            continuous[active]=target[active];remaining=0.
    quotas=np.minimum(np.floor(continuous+1e-10).astype(int),capacity)
    leftover=int(n-quotas.sum())
    if leftover<0:raise RuntimeError('Quota numerical overflow')
    order=np.argsort(-(continuous-quotas),kind='stable')
    for j in order:
        if leftover and quotas[j]<capacity[j]:quotas[j]+=1;leftover-=1
    if leftover or quotas.sum()!=n:raise RuntimeError('Quota allocation did not fill selection')
    return quotas


class CoverageSelector:
    def __init__(self,embeddings,reference_embeddings,props,reference_props,seed=42,property_weight=.5,marginal_weight=0.,redundancy_weight=0.):
        self.redundancy_weight=redundancy_weight;self.sequence_groups=None
        self.property_weight=property_weight;self.marginal_weight=marginal_weight
        if property_weight<0 or marginal_weight<0 or redundancy_weight<0:raise ValueError("Negative objective weight")
        self.seed=seed; self.e=np.asarray(embeddings); self.re=np.asarray(reference_embeddings)
        rp=np.asarray(reference_props); self.p=(np.asarray(props)-rp.mean(0))/rp.std(0).clip(.01)
        self.rp=(rp-rp.mean(0))/rp.std(0).clip(.01)
        pca=PCA(n_components=min(16,self.re.shape[1],len(self.re)),svd_solver='full')
        rz=pca.fit_transform(self.re); z=pca.transform(self.e)
        radius=max(float(np.sqrt(np.mean(np.sum((rz-rz.mean(0))**2,axis=1)))),1e-6)
        rz=rz/radius;z=z/radius
        joint_ref=np.column_stack([rz,.5*self.rp/2])
        joint=np.column_stack([z,.5*self.p/2])
        k=min(48,max(2,len(rp)//40),len(rp))
        self.km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(joint_ref)
        self.bins=self.km.predict(joint); self.weights=np.bincount(self.km.labels_,minlength=k)/len(rp)
        self.capacity=np.bincount(self.bins,minlength=k)
        ef=rff_fit(self.re,seed); pf=rff_fit(self.rp,seed+1)
        self.phi=np.column_stack([rff(self.e,ef),np.sqrt(property_weight)*rff(self.p,pf)])
        self.target=np.column_stack([rff(self.re,ef),np.sqrt(property_weight)*rff(self.rp,pf)]).mean(0)

        if marginal_weight:
            # Empirical CDF features match each descriptor distribution directly.
            thresholds=np.quantile(self.rp,np.linspace(.1,.9,9),axis=0)
            feature=lambda x: np.sqrt(marginal_weight/36)*(x[:,None,:]<=thresholds[None,:,:]).reshape(len(x),-1)
            self.phi=np.column_stack([self.phi,feature(self.p)])
            self.target=np.concatenate([self.target,feature(self.rp).mean(0)])

    def choose(self,seqs,utility,n,activity_weight=.005,iterations=250):
        if len(set(seqs))!=len(seqs): raise ValueError('Deduplicate candidate pool before selecting')
        utility=np.asarray(utility,dtype=float)
        if len(utility)!=len(seqs) or not np.isfinite(utility).all(): raise ValueError('Invalid utility')
        if self.redundancy_weight and self.sequence_groups is None:
            from rapidfuzz import process,fuzz
            representatives=[];labels=np.zeros(len(seqs),dtype=int)
            for i in sorted(range(len(seqs)),key=lambda j:seqs[j]):
                found=process.extractOne(seqs[i],representatives,scorer=fuzz.ratio,score_cutoff=80)
                if found is None:
                    labels[i]=len(representatives);representatives.append(seqs[i])
                else:labels[i]=found[2]
            self.sequence_groups=labels
        labels=self.sequence_groups if self.redundancy_weight else np.arange(len(seqs))
        quotas=allocate(n,self.weights,self.capacity)
        selected=[]
        for j,q in enumerate(quotas):
            ids=np.flatnonzero(self.bins==j)
            order=sorted(ids,key=lambda i:(-float(utility[i]),seqs[i]))
            selected.extend(order[:q])
        mask=np.zeros(len(seqs),bool);mask[selected]=True
        delta=self.phi[selected].mean(0)-self.target
        counts=np.bincount(labels[selected],minlength=int(labels.max())+1)
        penalty=lambda: self.redundancy_weight*float(np.sum(counts*(counts-1)))/n
        objective=lambda d,u: float(d@d-activity_weight*u+penalty())
        mean_u=float(utility[selected].mean()); initial=objective(delta,mean_u)
        rng=np.random.default_rng(self.seed); accepted=0
        members=[set(np.flatnonzero(mask & (self.bins==j))) for j in range(len(quotas))]
        for _ in range(iterations):
            ins=rng.integers(len(seqs),size=256);ins=ins[~mask[ins]]
            if not len(ins):break
            outs=[]; valid=[]
            # Swaps preserve development-region quotas exactly.
            for i in ins:
                eligible=sorted(members[self.bins[i]])
                if len(eligible):valid.append(i);outs.append(rng.choice(eligible))
            if not valid:continue
            ins=np.asarray(valid);outs=np.asarray(outs)
            shift=(self.phi[ins]-self.phi[outs])/n
            change=2*shift@delta+np.sum(shift*shift,axis=1)-activity_weight*(utility[ins]-utility[outs])/n
            if self.redundancy_weight:
                different=labels[ins]!=labels[outs]
                change+=self.redundancy_weight*2*(counts[labels[ins]]-counts[labels[outs]]+1)*different/n
            best=int(np.argmin(change))
            if change[best]<-1e-12:
                i,o=ins[best],outs[best];mask[o]=False;mask[i]=True
                counts[labels[o]]-=1;counts[labels[i]]+=1
                members[self.bins[o]].remove(int(o));members[self.bins[i]].add(int(i))
                delta+=shift[best];mean_u+=(utility[i]-utility[o])/n;accepted+=1
        chosen=sorted(np.flatnonzero(mask),key=lambda i:seqs[i])
        return chosen,{'objective_before':initial,'objective_after':objective(delta,mean_u),
            'selected_sequence_cluster_penalty':penalty(),'sequence_cluster_weight':self.redundancy_weight,'sequence_clusters':'Deterministic greedy Indel-ratio >=0.8 representative groups; heuristic, not all-pairs or official clustering','accepted_swaps':accepted,'activity_weight':activity_weight,'embedding_rff_dimensions':128,
            'property_rff_dimensions':128,'property_mmd_weight':self.property_weight,'marginal_cdf_weight':self.marginal_weight,'cluster_count':len(self.weights),
            'uncovered_reference_mass':float(self.weights[self.capacity==0].sum()),
            'quota_l1_difference':float(np.abs(quotas/n-self.weights).sum()),
            'development_region_weights':self.weights.tolist(),'candidate_region_counts':self.capacity.tolist(),
            'desired_uncapped_region_quotas':allocate(n,self.weights,np.full(len(self.weights),n)).tolist(),
            'region_quotas':quotas.tolist(),'selected_region_counts':np.bincount(self.bins[chosen],minlength=len(quotas)).tolist(),
            'warning':'Development surrogate objective; no hidden Phase-1 weights or global optimum claim.'}
