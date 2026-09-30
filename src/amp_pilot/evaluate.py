from pathlib import Path
import csv
import json
import sqlite3
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from rapidfuzz import process, fuzz
import torch
from .common import properties, features, read_fasta, write_fasta, json_write, device_for, sid

class ESMCache:
    def __init__(self,out,args):
        self.db=sqlite3.connect(Path(out)/'esm2_cache.sqlite'); self.model=None
        self.args=args; self.name=args.esm_model
        self.db.execute('CREATE TABLE IF NOT EXISTS embedding (model TEXT, sequence TEXT, vector BLOB, PRIMARY KEY(model,sequence))')
    def embed(self,seqs):
        missing=[]; cached={}
        for s in dict.fromkeys(seqs):
            r=self.db.execute('SELECT vector FROM embedding WHERE model=? AND sequence=?',(self.name,s)).fetchone()
            if r: cached[s]=np.frombuffer(r[0],dtype=np.float32)
            else: missing.append(s)
        if missing:
            if self.model is None:
                import esm
                loader=getattr(esm.pretrained,self.name)
                self.model,self.alphabet=loader(); self.model=self.model.to(device_for(self.args.device)).eval()
                self.convert=self.alphabet.get_batch_converter()
            device=device_for(self.args.device); batch=self.args.esm_batch
            for start in range(0,len(missing),batch):
                chunk=missing[start:start+batch]
                _,_,tok=self.convert([(str(i),s) for i,s in enumerate(chunk)])
                with torch.inference_mode():
                    layer=self.model.num_layers
                    rep=self.model(tok.to(device),repr_layers=[layer],return_contacts=False)['representations'][layer]
                    vectors=[rep[i,1:1+len(s)].mean(0).float().cpu().numpy() for i,s in enumerate(chunk)]
                for s,v in zip(chunk,vectors):
                    cached[s]=v; self.db.execute('INSERT OR REPLACE INTO embedding VALUES (?,?,?)',(self.name,s,v.tobytes()))
                self.db.commit()
                if start//batch%10==0: print(f'ESM2 cache {start+len(chunk)}/{len(missing)} new sequences',flush=True)
        return np.stack([cached[s] for s in seqs])
    def close(self): self.db.close()


def fbd(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if min(len(x),len(y))<2: return None
    a=np.atleast_2d(np.cov(x,rowvar=False)); b=np.atleast_2d(np.cov(y,rowvar=False))
    e,v=np.linalg.eigh((a+a.T)/2); sa=(v*np.sqrt(e.clip(0)))@v.T
    middle=sa@b@sa
    trace=np.sqrt(np.linalg.eigvalsh((middle+middle.T)/2).clip(0)).sum()
    return float(max(0,np.square(x.mean(0)-y.mean(0)).sum()+np.trace(a)+np.trace(b)-2*trace))


def mmd_rff(x,y,seed=42):
    # Biased approximate RBF MMD squared. Common reference and seed fix kernel across variants.
    rng=np.random.default_rng(seed); ref=y[:min(len(y),512)]
    dist=cdist(ref,ref,'sqeuclidean'); positive=dist[dist>0]
    bw=float(np.median(positive)) if len(positive) else 1.0
    w=rng.normal(size=(x.shape[1],256))/np.sqrt(max(bw,1e-8)); b=rng.uniform(0,2*np.pi,256)
    mx=np.cos(x@w+b).mean(0); my=np.cos(y@w+b).mean(0)
    return float((2/256)*np.square(mx-my).sum())


def precision_recall(x,y,k=3):
    # kNN-manifold estimate, exact within fixed-size samples only. Cubic-free, O(n*m) memory.
    if min(len(x),len(y))<=k: return {'precision':None,'recall':None}
    xx=cdist(x,x); yy=cdist(y,y); xy=cdist(x,y)
    rx=np.partition(xx,k,axis=1)[:,k]; ry=np.partition(yy,k,axis=1)[:,k]
    return {'precision':float((xy<=ry[None,:]).any(1).mean()),
            'recall':float((xy<=rx[:,None]).any(0).mean())}


def train_proxy(out,args):
    pos=read_fasta(out/'train.fasta'); neg=read_fasta(out/'train_negative.fasta')
    if len(neg)<20: return None,{'available':False,'reason':'Need >=20 labelled training negatives'}
    model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=3,class_weight='balanced',
                              n_jobs=args.workers,random_state=args.seed)
    model.fit(features(pos+neg),[1]*len(pos)+[0]*len(neg))
    ap=read_fasta(out/'audit.fasta'); an=read_fasta(out/'audit_negative.fasta')
    report={'available':True,'name':'local ExtraTrees AAC+dipeptide+properties',
            'warning':'Trained on the same positive corpus as generators; NOT independent competition surrogate or MIC model.',
            'train_pos':len(pos),'train_neg':len(neg),'audit_pos':len(ap),'audit_neg':len(an)}
    if ap and an:
        y=[1]*len(ap)+[0]*len(an); pred=model.predict_proba(features(ap+an))[:,1]
        report.update(audit_auroc=float(roc_auc_score(y,pred)),audit_average_precision=float(average_precision_score(y,pred)),audit_positive_prevalence=len(ap)/(len(ap)+len(an)))
    return model,report


def select(seqs,score,dev,n,seed):
    # Joint-property region quotas fitted to development set. Not a full Phase-1 optimizer.
    p=properties(dev); mu=p.mean(0); sd=p.std(0).clip(.01)
    k=min(24,max(2,len(dev)//20))
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit((p-mu)/sd)
    candidate_bins=km.predict((properties(seqs)-mu)/sd)
    weights=np.bincount(km.labels_,minlength=k)/len(dev)
    quotas=np.floor(n*weights).astype(int)
    order=np.argsort(-(n*weights-quotas),kind='stable'); quotas[order[:n-quotas.sum()]]+=1
    chosen=[]
    ranked=sorted(range(len(seqs)),key=lambda i:(-float(score[i]),seqs[i]))
    for region,q in enumerate(quotas): chosen += [i for i in ranked if candidate_bins[i]==region][:q]
    used=set(chosen); chosen += [i for i in ranked if i not in used][:n-len(chosen)]
    return chosen


def diversity(seqs,seed):
    if len(seqs)<2: return None
    rng=np.random.default_rng(seed); pairs=rng.integers(len(seqs),size=(min(4000,len(seqs)*4),2))
    pairs=pairs[pairs[:,0]!=pairs[:,1]]
    return float(np.mean([1-fuzz.ratio(seqs[a],seqs[b])/100 for a,b in pairs]))


def audit_set(seqs,ref,known,proxy,cache,args):
    rng=np.random.default_rng(args.seed+99)
    n=min(args.eval_n,len(seqs),len(ref))
    if n<4: return {'count':len(seqs),'insufficient_sample':True}
    # Uniform samples, equal-size within all metrics; reference order fixed across variants.
    x=[seqs[i] for i in rng.choice(len(seqs),n,replace=False)]
    r=[ref[i] for i in np.random.default_rng(args.seed+99).choice(len(ref),n,replace=False)]
    pp=properties(seqs); rp=properties(ref); sd=rp.std(0).clip(.01)
    result={'count':len(seqs),'sample_size':n,'mean_properties':pp.mean(0).tolist(),
            'property_wasserstein_std_units':[float(wasserstein_distance(pp[:,j],rp[:,j])/sd[j]) for j in range(4)],
            'joint_property_mmd_rff':mmd_rff(pp/sd,rp/sd,args.seed),
            'pairwise_indel_diversity':diversity(x,args.seed)}
    if proxy is not None:
        score=proxy.predict_proba(features(seqs))[:,1]
        result['local_activity_proxy_mean']=float(score.mean()); result['local_activity_proxy_q10']=float(np.quantile(score,.1))
    # Exact normalized Indel similarity on sampled sequences vs entire known-reference union.
    maxima=[]
    for start in range(0,len(x),64):
        d=process.cdist(x[start:start+64],known,scorer=fuzz.ratio,workers=args.workers,dtype=np.float32)
        maxima.extend((d.max(1)/100).tolist())
    result['known_max_ratio_median']=float(np.median(maxima))
    result['known_ratio_le_0_8_fraction']=float(np.mean(np.array(maxima)<=.8))
    if cache is not None:
        ex=cache.embed(x); er=cache.embed(r)
        result['esm2_fbd']=fbd(ex,er); result['esm2_mmd_rff']=mmd_rff(ex,er,args.seed)
        result.update({'esm2_'+k:v for k,v in precision_recall(ex,er).items()})
    return result


def evaluate(args,dirs):
    out=Path(args.out); ref=read_fasta(out/'audit.fasta'); dev=read_fasta(out/'dev.fasta')
    known=sorted(set(read_fasta(out/'all_positive.fasta'))|set(read_fasta(out/'forbidden.fasta',clean=False)))
    union=sorted({s for d in dirs for s in read_fasta(d/'eligible.fasta')})
    write_fasta(out/'scoring_input.fasta',union)
    proxy,proxy_report=train_proxy(out,args)
    cache=ESMCache(out,args) if args.esm else None
    report={'status':'PILOT_ONLY_NOT_OFFICIAL_PHASE1_SCORE','activity_proxy':proxy_report,
      'evaluation':{'esm2':args.esm_model if args.esm else None,'eval_n_cap':args.eval_n,
                    'property_definition':'HH pH7 fixed termini pKa; Eisenberg full-chain moment at 100 degrees',
                    'missing':['Independent AMP/MIC ensemble','ESM-C','official hidden references/weights','validated synthesizability'],
                    'warning':'Audit results become development evidence if repeatedly used to redesign; not an unbiased final test.'},
      'runs':{}}
    try:
        for dest in dirs:
            name=f'{dest.parent.name}/{dest.name}'; print('Evaluate '+name,flush=True)
            seqs=read_fasta(dest/'eligible.fasta',clean=False)
            scores=proxy.predict_proba(features(seqs))[:,1] if proxy and seqs else np.zeros(len(seqs))
            n=args.samples//4
            entry={'generation':json.loads((dest/'generation.json').read_text()),
                   'training':json.loads((dest.parent/'training_summary.json').read_text())}
            entry['raw_eligible']=audit_set(seqs,ref,known,proxy,cache,args)
            if len(seqs)>=n:
                chosen=select(seqs,scores,dev,n,args.seed)
                selected=[seqs[i] for i in chosen]; write_fasta(dest/'selected.fasta',selected)
                entry['selected']=audit_set(selected,ref,known,proxy,cache,args)
                top=np.argsort(-scores,kind='stable')[:n]
                entry['top_proxy_control']=audit_set([seqs[i] for i in top],ref,known,proxy,cache,args)
            else: entry['selected']={'insufficient_eligible':True,'required':n,'available':len(seqs)}
            with (dest/'candidates.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['sequence_id','sequence','local_activity_proxy'])
                for s,v in zip(seqs,scores): w.writerow([sid(s),s,float(v) if proxy else ''])
            report['runs'][name]=entry
            json_write(out/'report.json',report)
    finally:
        if cache is not None: cache.close()
    lines=['# AMP generator decision pilot','',
        '**No automatic Phase-1 winner is claimed.** This run measures engineering yield, local proxy scores and optional ESM2 distribution metrics.',
        '', '| Variant | Eligible / draws | Generation seconds | Selected local activity proxy | Selected ESM2 FBD (lower) | ESM2 precision | ESM2 recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,e in report['runs'].items():
        g=e['generation']; m=e.get('selected',{})
        def fmt(v): return f'{v:.4f}' if isinstance(v,(float,int)) else 'not measured'
        lines.append(f"| {name} | {g['unique_eligible']} / {g['draws']} | {g['seconds']:.1f} | {fmt(m.get('local_activity_proxy_mean'))} | {fmt(m.get('esm2_fbd'))} | {fmt(m.get('esm2_precision'))} | {fmt(m.get('esm2_recall'))} |")
    lines += ['', '## Interpretation', '',
        '- Compare yield, sampled novelty, coverage and selected-set fidelity together. MSE and cross entropy are not comparable.',
        '- Review `training.jsonl`: the time cap may stop models at different update counts. No equal-convergence claim.',
        '- AR sampling recomputes prefixes; speed results do not characterize an optimized KV-cached implementation.',
        '- The local activity proxy is trained here and shares the generator positive corpus. It is not an independent potency assessment.',
        '- A 5k pilot cannot prove 50k unique yield. Run a larger pool only after this diagnostic passes.',
        '- Missing MIC, ESM-C and official metrics remain missing; do not infer them from ESM2 or properties.',
        '', '## Files to share', '', '`report.json`, `report.md`, and both model `training_summary.json` files.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(f'Finished: {out / "report.md"}',flush=True)
    return report
