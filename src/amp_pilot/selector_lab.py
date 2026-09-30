"""Scored-pool selector comparison; never generates or runs BATTLE."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix,eye
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from .common import read_fasta,write_fasta,digest,json_write
from .production import csv_rows,activity_summary,fingerprint,valid
from .production_select import GuardedSelector,max_similarity,neighbor_stats,potency_utility
from .phase1 import selection_properties,seqme_properties,chemistry_diagnostics,embedding_metrics
from .evaluate import ESMCache,fbd
from types import SimpleNamespace

NAMES=['length','charge','hydrophobicity','hydrophobic_moment']

def allocate_cells(labels,hit,quality,n,region,props,ref,region_weights,mode,target=.8,embedding_features=None,embedding_target=None):
    """LP cell counts with regional/CDF balance; final exact metrics are separately required.

    Each cell includes a hit/nonhit flag. Ranking within a cell prefers classifier,
    MIC and novelty utility. LP balances counts, not fractional peptide identities.
    """
    keys=np.column_stack([labels,hit.astype(int)])
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    groups=[[] for _ in range(inverse.max()+1)]
    for i,j in enumerate(inverse):groups[j].append(i)
    groups=[np.array(sorted(g,key=lambda i:(-quality[i],i)),dtype=np.int64) for g in groups]
    representatives=np.array([g[0] for g in groups]);cap=np.array([len(g) for g in groups]);g=len(groups)
    # Regional membership and descriptor threshold membership are constant inside cells.
    features=[];targets=[];weights=[]
    if mode in ['regional','hybrid','balanced80','moment80']:
        for j,w in enumerate(region_weights):
            features.append((region[representatives]==j).astype(float));targets.append(w);weights.append(1.)
    if mode in ['properties','hybrid','balanced80','moment80']:
        for j in range(4):
            edges=np.unique(np.quantile(ref[:,j],np.linspace(0,1,13)))
            widths=np.diff(edges)/max(ref[:,j].std(),.01)
            for t,width in zip(edges[:-1],widths):
                features.append((props[representatives,j]<=t).astype(float));targets.append(np.mean(ref[:,j]<=t));weights.append(float(width))
    if mode=='moment80':
        if embedding_features is None or embedding_target is None:raise ValueError('Embedding moments required')
        for j,t in enumerate(embedding_target):
            features.append(np.array([embedding_features[members,j].mean() for members in groups]));targets.append(float(t));weights.append(.25)
    F=np.asarray(features);b=np.asarray(targets)*n;w=np.asarray(weights);k=len(b)
    # |F x - target*N| <= slack. Soft balance is unavoidable when cells lack supply;
    # audited limits below determine eligibility, rather than hiding infeasibility.
    A=vstack([hstack([csr_matrix(F),-eye(k)]),hstack([csr_matrix(-F),-eye(k)])],format='csr')
    rhs=np.r_[b,-b]
    hitcell=hit[representatives].astype(float)
    balance={'regional':12.,'properties':12.,'hybrid':18.,'balanced80':18.,'moment80':18.}[mode]
    hit_reward=0. if mode in ['balanced80','moment80'] else 1.
    c=np.r_[-hit_reward*hitcell-.02*np.array([quality[x].mean() for x in groups]),balance*w]
    equal=csr_matrix(np.r_[np.ones(g),np.zeros(k)][None,:])
    bounds=[(0,float(x)) for x in cap]+[(0,None)]*k
    # Try explicit 80% constraint first, then maximize achievable hit count with balance.
    targetrow=csr_matrix(np.r_[-hitcell,np.zeros(k)][None,:])
    result=linprog(c,A_ub=vstack([A,targetrow]),b_ub=np.r_[rhs,-np.ceil(target*n)],A_eq=equal,b_eq=[n],bounds=bounds,method='highs')
    target_lp_feasible=result.success
    if not result.success:
        if result.status!=2:raise RuntimeError('Allocation solver failed: '+result.message)
        result=linprog(c,A_ub=A,b_ub=rhs,A_eq=equal,b_eq=[n],bounds=bounds,method='highs')
    if not result.success:raise RuntimeError('Allocation solver failed: '+result.message)
    x=np.clip(result.x[:g],0,cap);counts=np.floor(x+1e-7).astype(int)
    # Exact size after integer rounding. Prefer hit cells among fractional allocations.
    deficit=n-int(counts.sum())
    if deficit<0:raise RuntimeError('LP rounding exceeded requested size')
    order=sorted(range(g),key=lambda j:(-hitcell[j],-(x[j]-counts[j]),-quality[groups[j][0]],j))
    for j in order:
        if deficit==0:break
        if counts[j]<cap[j] and x[j]-counts[j]>1e-6:counts[j]+=1;deficit-=1
    if deficit:
        for j in order:
            add=min(deficit,int(cap[j]-counts[j]));counts[j]+=add;deficit-=add
            if not deficit:break
    if deficit:raise RuntimeError('Insufficient cell capacity')
    ids=np.concatenate([members[:count] for members,count in zip(groups,counts) if count])
    return ids,{'cells':g,'lp_target_feasible':bool(target_lp_feasible),'lp_message':result.message,'lp_hit_fraction':float(hitcell@x/n),'rounded_hit_fraction':float(hit[ids].mean()),'note':'LP distribution terms use threshold CDF proxies; final exact distances and neighbors determine acceptance.'}


def repair(seqs,ids,order,values,workers,budget,repair_labels=None):
    """Replace connected sequences only with zero-edge incoming candidates.

    Never reduces hit count; never increases sum MIC. Final distribution checks
    are mandatory because these swaps can change property/embedding balance.
    """
    from rapidfuzz import process,fuzz
    ids=ids.copy();chosen=[seqs[i] for i in ids];before,counts=neighbor_stats(chosen,workers,return_counts=True)
    selected=set(map(int,ids));fixed=0;attempts=0
    for i in order:
        if attempts>=budget or not counts.any():break
        i=int(i)
        if i in selected:continue
        attempts+=1
        eligible=np.flatnonzero((counts>0)&(values[ids,-1]>=values[i,-1]))
        if repair_labels is not None:eligible=eligible[repair_labels[ids[eligible]]==repair_labels[i]]
        if not len(eligible):continue
        slot=int(eligible[np.lexsort((-values[ids[eligible],-1],-counts[eligible]))[0]])
        o=int(ids[slot]);incoming=process.cdist([seqs[i]],chosen,scorer=fuzz.ratio,workers=workers,dtype=np.float32)[0];incoming[slot]=0
        if np.any(incoming>80):continue
        old=process.cdist([seqs[o]],chosen,scorer=fuzz.ratio,workers=workers,dtype=np.float32)[0]>80;old[slot]=False
        counts[old]-=1;counts[slot]=0;selected.remove(o);selected.add(i);ids[slot]=i;chosen[slot]=seqs[i];fixed+=1
    return ids,{'before':before,'after':neighbor_stats(chosen,workers),'accepted':fixed,'attempted':attempts}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',default='runs/production_v1');p.add_argument('--pilot',default='runs/pilot')
    p.add_argument('--out',default='runs/selector_lab_v1');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--size',type=int,default=50000);p.add_argument('--target',type=float,default=.8)
    p.add_argument('--reuse-comparison',help='Reuse validated known-similarity and baseline metrics from an earlier comparison')
    p.add_argument('--repair-budget',type=int,default=3000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--policies',nargs='+',choices=['regional','properties','hybrid','balanced80','moment80'],default=['regional','properties','hybrid'])
    a=p.parse_args(argv)
    if a.size<4 or not 0<a.target<=1 or a.workers<1 or a.repair_budget<0:p.error('Invalid arguments')
    with threadpool_limits(limits=a.workers):run(a)


def run(a):
    root=Path(a.production).resolve();pilot=Path(a.pilot).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    seqs=read_fasta(root/'candidates.fasta',False);valid(seqs,'pool');table=csv_rows(root/'scores.csv')
    if set(table)!=set(seqs) or a.size>len(seqs):raise ValueError('Pool/score mismatch or insufficient candidates')
    dev=read_fasta(pilot/'dev.fasta',False);audit=read_fasta(pilot/'audit.fasta',False)
    if set(dev)&set(audit):raise ValueError('Development/audit overlap')
    known=sorted(set(read_fasta(pilot/'all_positive.fasta'))|set(read_fasta(pilot/'forbidden.fasta',False)))
    known=[s for s in known if s and set(s)<=set('ACDEFGHIKLMNPQRSTVWY')]
    if set(seqs)&set(known):raise ValueError('Pool contains excluded known sequences')
    # Baseline used in the previous run, not a newly drawn easier comparator.
    baseline_seq=read_fasta(root/'baseline.fasta',False);index={s:i for i,s in enumerate(seqs)}
    if len(baseline_seq)!=a.size or any(s not in index for s in baseline_seq):raise ValueError('Baseline mismatch; --size must match production baseline')
    baseline=np.array([index[s] for s in baseline_seq]);values=np.stack([table[s] for s in seqs])
    signature={'args':vars(a),'pool':digest(root/'candidates.fasta'),'scores':digest(root/'scores.csv'),'baseline':digest(root/'baseline.fasta'),'dev':digest(pilot/'dev.fasta'),'audit':digest(pilot/'audit.fasta'),'known':fingerprint(known),'code':digest(__file__)}
    pin=out/'manifest.json'
    if pin.exists() and json.loads(pin.read_text())!=signature:raise RuntimeError('Inputs/settings/code changed. Choose a new --out; existing comparison preserved.')
    json_write(pin,signature)
    cache=ESMCache(pilot,SimpleNamespace(device=a.device,esm_model='esm2_t6_8M_UR50D',esm_batch=32))
    try:
        props=selection_properties(seqs);rp=selection_properties(dev);e=cache.embed(seqs);de=cache.embed(dev)
        cover=GuardedSelector(e,de,props,rp,a.seed).cover
        if a.reuse_comparison:
            import shutil
            previous=Path(a.reuse_comparison).resolve();prev=json.loads((previous/'manifest.json').read_text())
            for key in ['pool','scores','baseline','dev','audit','known']:
                if prev.get(key)!=signature[key]:raise ValueError('Reuse input mismatch: '+key)
            if prev['args']['seed']!=a.seed or prev['args']['size']!=a.size:raise ValueError('Reuse seed/size mismatch')
            for name in ['known_similarity.npy','baseline_metrics.json']:
                if not (out/name).exists():shutil.copyfile(previous/name,out/name)
        kp=out/'known_similarity.npy' 
        if kp.exists():similarity=np.load(kp,allow_pickle=False)
        else:
            similarity=max_similarity(seqs,known,a.workers);np.save(kp,similarity)
        if similarity.shape!=(len(seqs),) or not np.isfinite(similarity).all():raise ValueError('Invalid similarity cache')
        hit=values[:,-1]<=16;quality=potency_utility(values,similarity)
        # Cell membership includes EVERY threshold used by the LP, so histogram
        # contributions are identical for all members, not representative estimates.
        codes=[]
        for j in range(4):
            edges=np.unique(np.quantile(rp[:,j],np.linspace(0,1,13)))[:-1]
            codes.append(np.searchsorted(edges,props[:,j],side='left'))
        allcodes=np.column_stack(codes)
        def metrics(ids,neighbors=None):
            ss=[seqs[i] for i in ids]
            return {'activity':activity_summary(values[ids]),'wasserstein':{name:float(wasserstein_distance(props[ids,j],rp[:,j])/max(rp[:,j].std(),.01)) for j,name in enumerate(NAMES)},'fbd_development':float(fbd(e[ids],de)),'conformity_development':float(seqme_properties(ss,dev)['conformity_mean']),'known_mean':float(similarity[ids].mean()),'known_p90':float(np.quantile(similarity[ids],.9)),'known_fraction_le_0_8':float((similarity[ids]<=.8).mean()),'neighbors':neighbors if neighbors is not None else neighbor_stats(ss,a.workers),'region_l1':float(np.abs(np.bincount(cover.bins[ids],minlength=len(cover.weights))/len(ids)-cover.weights).sum())}
        bp=out/'baseline_metrics.json'
        b=json.loads(bp.read_text()) if bp.exists() else metrics(baseline);json_write(bp,b)
        # Fixed, disclosed engineering allowances; no audit-set tuning.
        limits={'wasserstein':{k:max(v*1.1,v+.025) for k,v in b['wasserstein'].items()},'fbd':max(b['fbd_development']*1.1,b['fbd_development']+.01),'conformity_min':b['conformity_development']-.02,'region_l1_max':b['region_l1']+.03}
        def failures(m):
            f=[k+'_wasserstein' for k in NAMES if m['wasserstein'][k]>limits['wasserstein'][k]+1e-9]
            if m['fbd_development']>limits['fbd']:f.append('embedding_fbd')
            if m['conformity_development']<limits['conformity_min']:f.append('conformity')
            if m['region_l1']>limits['region_l1_max']:f.append('regional_coverage')
            if m['known_mean']>b['known_mean']+1e-6 or m['known_p90']>b['known_p90']+.01 or m['known_fraction_le_0_8']<b['known_fraction_le_0_8']-.005:f.append('novelty')
            if m['activity']['amplify_mean']<b['activity']['amplify_mean']-.01:f.append('amplify')
            for k in ['predicted_mic_um_mean','predicted_mic_um_median','predicted_mic_um_p90']:
                if m['activity'][k]>b['activity'][k]+1e-8:f.append(k)
            for k in ['near_neighbor_fraction_gt_0_8','mean_neighbors_gt_0_8']:
                if m['neighbors'][k]>b['neighbors'][k]+1e-9:f.append(k)
            if m['neighbors']['nearest_distance_q10']<b['neighbors']['nearest_distance_q10']-.01:f.append('nearest_distance_q10')
            return f
        # Reference-fitted PCA mean/covariance surrogate. Full-space FBD remains
        # the final acceptance test; cell means are only allocation approximations.
        from sklearn.decomposition import PCA
        pc=PCA(n_components=min(8,de.shape[1],len(de)),svd_solver='full').fit(de)
        scale=np.sqrt(pc.explained_variance_).clip(1e-3)
        z=pc.transform(e)/scale;zr=pc.transform(de)/scale
        pairs=[(i,j) for i in range(z.shape[1]) for j in range(i,z.shape[1])]
        ef=np.column_stack([z,*[z[:,i]*z[:,j] for i,j in pairs]])
        et=np.column_stack([zr,*[zr[:,i]*zr[:,j] for i,j in pairs]]).mean(0)
        results={}
        for policy in dict.fromkeys(a.policies):
            folder=out/policy;folder.mkdir(exist_ok=True);record=folder/'report.json'
            if record.exists():
                saved=json.loads(record.read_text())
                if digest(folder/'library.fasta')!=saved['fasta_sha256']:raise RuntimeError('Changed selector output')
                results[policy]=saved;continue
            print('SELECTOR',policy,flush=True)
            # Region-only still includes descriptor bins to keep cells consistent
            # across approaches and enable a fair allocation comparison.
            labels=np.column_stack([cover.bins,allcodes])
            ids,allocation=allocate_cells(labels,hit,quality,a.size,cover.bins,props,rp,cover.weights,policy,a.target,ef,et)
            np.save(folder/'allocated_indices.npy',ids)
            repair_labels=None
            if policy in ['balanced80','moment80']:
                # Preserve embedding-region and all descriptor histogram counts.
                _,repair_labels=np.unique(labels,axis=0,return_inverse=True)
            ids,rep=repair(seqs,ids,np.argsort(-quality,kind='stable'),values,a.workers,a.repair_budget,repair_labels)
            allocation['threshold_reward_above_target']=policy not in ['balanced80','moment80']
            allocation['repair_preserves_region_and_descriptor_cells']=repair_labels is not None
            np.save(folder/'selected_indices.npy',ids)
            m=metrics(ids,rep['after']);failed=failures(m);ss=sorted(seqs[i] for i in ids);write_fasta(folder/'library.fasta',ss)
            # Held-out diagnostic only; never feeds selection or recommendation.
            rng=np.random.default_rng(a.seed+811);sample=[ss[i] for i in rng.choice(len(ss),min(5000,len(ss)),replace=False)]
            held={'properties':seqme_properties(ss,audit),'esm2':embedding_metrics(cache.embed(sample),cache.embed(audit),de,1000,3,a.seed+811),'chemistry':chemistry_diagnostics(ss)}
            result={'policy':policy,'allocation':allocation,'repair':rep,'metrics':m,'limits':limits,'failed_guards':failed,'balance_passed':not failed,'potency_target_met':m['activity']['predicted_fraction_le_16uM']>=a.target,'held_out_audit':held,'fasta_sha256':digest(folder/'library.fasta'),'warning':'Predicted MIC only; no ESM-C, MMseqs, safety certification, or final top-100 eligibility certification.'}
            json_write(record,result);results[policy]=result
            print(policy,'MIC<=16',m['activity']['predicted_fraction_le_16uM'],'failed',failed,flush=True)
        eligible=[k for k,v in results.items() if v['balance_passed']]
        winner=max(eligible,key=lambda k:(results[k]['potency_target_met'],results[k]['metrics']['activity']['predicted_fraction_le_16uM'],-results[k]['metrics']['activity']['predicted_mic_um_median'])) if eligible else None
        summary={'complete':True,'target':a.target,'baseline':b,'limits':limits,'results':results,'recommended_policy':winner,'target_achieved_with_balance':bool(winner and results[winner]['potency_target_met']),'pool_low_mic_count':int(hit.sum()),'note':'No production files replaced. No feasible challenger means no recommendation. Missing target is explicit. Held-out audit is diagnostic only.'}
        json_write(out/'comparison.json',summary)
        lines=['# Selector comparison','','| Policy | MIC <=16 | Median MIC | Balance passed | Target met | Failed checks |','|---|---:|---:|---|---|---|']
        for k,v in results.items():lines.append(f"| {k} | {v['metrics']['activity']['predicted_fraction_le_16uM']:.2%} | {v['metrics']['activity']['predicted_mic_um_median']:.3f} | {v['balance_passed']} | {v['potency_target_met']} | {', '.join(v['failed_guards'])} |")
        lines+=['',f'Recommended policy: {winner or "none"}.','',summary['note']];(out/'comparison.md').write_text('\n'.join(lines)+'\n')
        print('SELECTOR COMPARISON COMPLETE',out,'recommended:',winner,flush=True)
    finally:cache.close()

if __name__=='__main__':main()
