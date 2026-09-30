"""Full-space FBD-gradient exchange and compensated pair search from a saved hybrid.
No generation or predictor inference. Exact transactional checkpoint acceptance.
"""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from .common import read_fasta,write_fasta,json_write,digest
from .production import csv_rows,activity_summary,fingerprint,valid
from .production_select import neighbor_stats,max_similarity
from .selector import CoverageSelector
from .phase1 import selection_properties,seqme_properties,embedding_metrics,chemistry_diagnostics
from .evaluate import ESMCache,fbd

NAMES=['length','charge','hydrophobicity','hydrophobic_moment']


class GaussianState:
    """Sufficient statistics for exact Gaussian FBD after one or two swaps."""
    def __init__(self,e,ref,ids):
        x=np.asarray(e[ids],float);r=np.asarray(ref,float);self.n=len(x)
        self.total=x.sum(0);self.second=x.T@x;self.refmean=r.mean(0)
        B=np.atleast_2d(np.cov(r,rowvar=False));v,U=np.linalg.eigh((B+B.T)/2)
        self.refsqrt=(U*np.sqrt(v.clip(0)))@U.T;self.reftrace=float(np.trace(B))
    def covariance(self,total=None,second=None):
        t=self.total if total is None else total;q=self.second if second is None else second
        return (q-np.outer(t,t)/self.n)/(self.n-1)
    def distance(self,total=None,second=None):
        t=self.total if total is None else total;A=self.covariance(total,second)
        middle=self.refsqrt@A@self.refsqrt
        trace=np.sqrt(np.linalg.eigvalsh((middle+middle.T)/2).clip(0)).sum()
        return float(max(0.,np.square(t/self.n-self.refmean).sum()+np.trace(A)+self.reftrace-2*trace))
    def proposal(self,e,outgoing,incoming):
        old=np.asarray(e[outgoing],float);new=np.asarray(e[incoming],float)
        return self.total+new.sum(0)-old.sum(0),self.second+new.T@new-old.T@old


def fbd_gradient(e,ref,ids,block=2048,state=None):
    """Gradient of Gaussian squared Wasserstein distance, in original ESM space.

    A tiny PSD regularizer stabilizes the gradient only. Exact acceptance uses
    the unregularized full-space FBD from evaluate.py.
    """
    x=np.asarray(e[ids],float);r=np.asarray(ref,float);mu=x.mean(0);mr=r.mean(0)
    A=state.covariance() if state is not None else np.atleast_2d(np.cov(x,rowvar=False));B=np.atleast_2d(np.cov(r,rowvar=False))
    ev,U=np.linalg.eigh((A+A.T)/2);eps=max(1e-8,float(np.trace(A)/len(A))*1e-6)
    sa=(U*np.sqrt(ev.clip(eps)))@U.T;inv=(U*(1/np.sqrt(ev.clip(eps))))@U.T
    middle=sa@B@sa;v,V=np.linalg.eigh((middle+middle.T)/2)
    middle_sqrt=(V*np.sqrt(v.clip(0)))@V.T
    G=np.eye(len(A))-inv@middle_sqrt@inv;G=(G+G.T)/2
    output=np.empty(len(e))
    for i in range(0,len(e),block):
        z=np.asarray(e[i:i+block],float)-mu
        output[i:i+len(z)]=np.einsum('ij,ij->i',z@G,z)+2*z@(mu-mr)
    return output


class WassersteinPotential:
    """Exact 1-D Wasserstein distances and a CDF subgradient at current selection."""
    def __init__(self,props,reference):
        self.columns=[];self.scale=reference.std(0).clip(.01)
        for j in range(props.shape[1]):
            support=np.unique(np.r_[props[:,j],reference[:,j]])
            indices=np.searchsorted(support,props[:,j]);ri=np.searchsorted(support,reference[:,j])
            refcdf=np.cumsum(np.bincount(ri,minlength=len(support)))/len(ri)
            self.columns.append((support,indices,refcdf))
    def evaluate(self,ids):
        distances=[];pot=np.zeros((len(self.columns[0][1]),len(self.columns)))
        for j,(support,indices,rcdf) in enumerate(self.columns):
            cdf=np.cumsum(np.bincount(indices[ids],minlength=len(support)))/len(ids)
            delta=cdf[:-1]-rcdf[:-1];width=np.diff(support)/self.scale[j]
            distances.append(float(np.abs(delta)@width))
            integral=np.r_[np.cumsum((np.sign(delta)*width)[::-1])[::-1],0.]
            pot[:,j]=integral[indices]
        return np.array(distances),pot


def exact_metrics(ids,e,de,props,rp,values,similarity,bins,weights,fbd_override=None):
    return {'fbd':float(fbd(e[ids],de)) if fbd_override is None else float(fbd_override),
        'wasserstein':{k:float(wasserstein_distance(props[ids,j],rp[:,j])/max(rp[:,j].std(),.01)) for j,k in enumerate(NAMES)},
        'region_l1':float(np.abs(np.bincount(bins[ids],minlength=len(weights))/len(ids)-weights).sum()),
        'activity':activity_summary(values[ids]),'known_mean':float(similarity[ids].mean()),
        'known_p90':float(np.quantile(similarity[ids],.9)),'known_fraction_le_0_8':float(np.mean(similarity[ids]<=.8))}


def violation(m,limits):
    terms=[max(0.,m['fbd']/limits['fbd']-1),max(0.,m['region_l1']/limits['region_l1_max']-1)]
    terms.extend(max(0.,m['wasserstein'][k]/limits['wasserstein'][k]-1) for k in NAMES)
    return float(sum(terms))


def safe_transition(old,new,start,limits,floor):
    """No passing guard can become failing. Existing violations cannot grow."""
    if new['activity']['predicted_fraction_le_16uM']+1e-12<floor:return False
    if new['known_mean']>start['known_mean']+1e-6 or new['known_p90']>start['known_p90']+.005 or new['known_fraction_le_0_8']<start['known_fraction_le_0_8']-.002:return False
    if new['activity']['amplify_mean']<start['activity']['amplify_mean']-.01:return False
    for k in ['predicted_mic_um_mean','predicted_mic_um_median','predicted_mic_um_p90']:
        if new['activity'][k]>start['activity'][k]*1.03+1e-8:return False
    if new['fbd']>max(limits['fbd'],old['fbd'])+1e-9 or new['region_l1']>max(limits['region_l1_max'],old['region_l1'])+1e-9:return False
    for k in NAMES:
        if new['wasserstein'][k]>max(limits['wasserstein'][k],old['wasserstein'][k])+1e-9:return False
    return True


def refine(seqs,ids,e,de,props,rp,values,similarity,bins,weights,limits,floor,mode,epochs,workers,seed,checkpoint=None,original_guard=None):
    """Exchange heuristic with exact global checks and actual sequence comparisons.

    Shortlisted proposals use full-FBD gradient plus exact CDF potentials.
    Pair mode also searches a compensating exchange for the descriptor delta.
    Neither procedure is a proof of a globally optimal selection.
    """
    from rapidfuzz import process,fuzz
    rng=np.random.default_rng(seed);ids=ids.copy();n=len(ids)
    start=exact_metrics(ids,e,de,props,rp,values,similarity,bins,weights);current=start
    guard=original_guard if original_guard is not None else start
    wp=WassersteinPotential(props,rp);trace=[];accepted=0;scales=rp.std(0).clip(.01)
    def objective(m):return 100*violation(m,limits)+m['fbd']+.5*m['region_l1']
    for epoch in range(epochs):
        state=GaussianState(e,de,ids)
        grad=fbd_gradient(e,de,ids,state=state);dist,potential=wp.evaluate(ids)
        counts=np.bincount(bins[ids],minlength=len(weights))/n
        region_gradient=np.sign(counts-weights)[bins]
        # Focus on whichever guard remains violated; use property slack to
        # prevent gradients from moving already balanced tails gratuitously.
        property_weight=np.array([2. if dist[j]>limits['wasserstein'][k]*.8 else .25 for j,k in enumerate(NAMES)])
        fbd_weight=1.+(100./limits['fbd'] if current['fbd']>limits['fbd'] else 0.)
        region_weight=.5+(100./limits['region_l1_max'] if current['region_l1']>limits['region_l1_max'] else 0.)
        score=fbd_weight*grad+region_weight*region_gradient+potential@property_weight+.05*similarity
        selected=np.zeros(len(seqs),bool);selected[ids]=True
        # Include useful high-potency replacements and a deterministic random
        # exploration sample; no hidden held-out feedback.
        candidates=np.flatnonzero(~selected)
        order=candidates[np.argsort(score[candidates],kind='stable')[:512]]
        if len(candidates):order=np.unique(np.r_[order,rng.choice(candidates,min(128,len(candidates)),replace=False)])
        order=order[np.argsort(score[order],kind='stable')]
        slots=np.argsort(-score[ids],kind='stable')[:256]
        epoch_accepted=0;proposals=0
        for incoming in order:
            if epoch_accepted>=24:break
            incoming=int(incoming)
            if selected[incoming]:continue
            # Explore multiple outgoing members, including donors in surplus regions.
            eligible=[int(slot) for slot in slots if score[incoming]<score[ids[slot]]-1e-8]
            if not eligible:continue
            outslot=eligible[0];outgoing=int(ids[outslot]);moves=[(outslot,incoming)]
            if mode=='paired':
                delta=(props[incoming]-props[outgoing])/scales
                remaining=order[(order!=incoming)&(~selected[order])][:64]
                donors=np.array([slot for slot in eligible[1:33] if slot!=outslot],int)
                if len(remaining) and len(donors):
                    # Beam chooses complementary descriptor movement, with the
                    # same full-space gradient as a secondary criterion.
                    residual=delta[None,None,:]+(props[remaining,None,:]-props[ids[donors]][None,:,:])/scales
                    costs=np.square(residual).sum(2)+.05*(score[remaining,None]-score[ids[donors]][None,:])
                    j,k=np.unravel_index(np.argmin(costs),costs.shape)
                    moves.append((int(donors[k]),int(remaining[j])))
            proposed=ids.copy()
            for slot,i in moves:proposed[slot]=i
            if len(set(map(int,proposed)))!=n:continue
            if np.mean(values[proposed,-1]<=16)+1e-12<floor:continue
            proposals+=1
            if proposals%64==0:print(mode,'epoch',epoch+1,'proposals',proposals,'accepted',epoch_accepted,flush=True)
            # Incoming peptides create ZERO >80% Indel edges to the proposed
            # full library (including each other); old edges may only disappear.
            chosen=[seqs[i] for i in proposed];edge=False
            for slot,i in moves:
                similarities=process.cdist([seqs[i]],chosen,scorer=fuzz.ratio,workers=workers,dtype=np.float32)[0];similarities[slot]=0
                if np.any(similarities>80):edge=True;break
            if edge:continue
            total,second=state.proposal(e,[ids[slot] for slot,_ in moves],[i for _,i in moves])
            m=exact_metrics(proposed,e,de,props,rp,values,similarity,bins,weights,state.distance(total,second))
            if not safe_transition(current,m,guard,limits,floor) or objective(m)>=objective(current)-1e-9:continue
            for slot,i in moves:selected[ids[slot]]=False;selected[i]=True
            state.total=total;state.second=second
            ids=proposed;current=m;accepted+=len(moves);epoch_accepted+=1
        item={'epoch':epoch+1,'accepted_exchanges':epoch_accepted,'tested_edge_feasible_proposals':proposals,'metrics':current,'violation':violation(current,limits)}
        trace.append(item);print(mode,'epoch',epoch+1,'exchanges',epoch_accepted,'FBD',current['fbd'],'region',current['region_l1'],'MIC<=16',current['activity']['predicted_fraction_le_16uM'],flush=True)
        if checkpoint:checkpoint(ids,trace)
        if epoch_accepted==0 or violation(current,limits)<=1e-12:break
    return ids,{'initial':start,'final':current,'trace':trace,'accepted_sequence_replacements':accepted,'objective':'100*normalized_distribution_guard_violation + full_FBD + 0.5*regional_L1','warning':'Local heuristic; zero-edge insertion does not guarantee lower-distance quantiles, which are audited separately.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',default='runs/production_v1');p.add_argument('--pilot',default='runs/pilot')
    p.add_argument('--source',default='runs/selector_lab_v1');p.add_argument('--out',default='runs/hybrid_refine_v1')
    p.add_argument('--modes',nargs='+',choices=['exchange','paired'],default=['exchange','paired'])
    p.add_argument('--potency-floor',type=float,default=.88);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args(argv)
    if a.epochs<1 or a.workers<1 or not .8<=a.potency_floor<=1:p.error('Invalid settings; potency floor must be >=80%')
    with threadpool_limits(limits=a.workers):run(a)


def run(a):
    root=Path(a.production).resolve();pilot=Path(a.pilot).resolve();source=Path(a.source).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    seqs=read_fasta(root/'candidates.fasta',False);valid(seqs,'pool');table=csv_rows(root/'scores.csv')
    if set(table)!=set(seqs):raise ValueError('Scores/pool mismatch')
    dev=read_fasta(pilot/'dev.fasta',False);audit=read_fasta(pilot/'audit.fasta',False)
    if set(dev)&set(audit):raise ValueError('Reference overlap')
    previous=json.loads((source/'manifest.json').read_text());comparison=json.loads((source/'comparison.json').read_text());source_report=comparison['results']['hybrid']
    for key,path in [('pool',root/'candidates.fasta'),('scores',root/'scores.csv'),('dev',pilot/'dev.fasta'),('audit',pilot/'audit.fasta')]:
        if previous[key]!=digest(path):raise ValueError('Source cache/input mismatch: '+key)
    if previous['args']['seed']!=a.seed:raise ValueError('Use the original source seed for region consistency')
    hybrid=source/'hybrid/library.fasta'
    if digest(hybrid)!=source_report['fasta_sha256']:raise ValueError('Hybrid FASTA changed')
    index={s:i for i,s in enumerate(seqs)};ss=read_fasta(hybrid,False);valid(ss,'hybrid')
    ids=np.array([index[s] for s in ss]);values=np.stack([table[s] for s in seqs])
    if np.mean(values[ids,-1]<=16)<a.potency_floor:raise ValueError('Starting hybrid is below requested potency floor')
    known=sorted(set(read_fasta(pilot/'all_positive.fasta'))|set(read_fasta(pilot/'forbidden.fasta',False)))
    known=[s for s in known if s and set(s)<=set('ACDEFGHIKLMNPQRSTVWY')]
    if previous['known']!=fingerprint(known) or set(seqs)&set(known):raise ValueError('Known reference changed or forbidden pool member')
    similarity=np.load(source/'known_similarity.npy',allow_pickle=False)
    if similarity.shape!=(len(seqs),) or not np.isfinite(similarity).all():raise ValueError('Similarity cache invalid')
    signature={'args':vars(a),'source_fasta':digest(hybrid),'source_manifest':digest(source/'manifest.json'),'source_comparison':digest(source/'comparison.json'),'similarity':digest(source/'known_similarity.npy'),'runner':digest(__file__),'dependencies':{name:digest(Path(__file__).with_name(name)) for name in ['selector.py','production.py','phase1.py','evaluate.py','production_select.py']}}
    pin=out/'manifest.json'
    if pin.exists() and json.loads(pin.read_text())!=signature:raise ValueError('Changed inputs/code/settings; use a fresh --out')
    json_write(pin,signature)
    cache=ESMCache(pilot,SimpleNamespace(device=a.device,esm_model='esm2_t6_8M_UR50D',esm_batch=32))
    try:
        props=selection_properties(seqs);rp=selection_properties(dev);e=cache.embed(seqs);de=cache.embed(dev)
        cover=CoverageSelector(e,de,props,rp,a.seed);original_guard=exact_metrics(ids,e,de,props,rp,values,similarity,cover.bins,cover.weights);limits=comparison['limits'];baseline=comparison['baseline'];source_neighbors=source_report['metrics']['neighbors'];results={}
        for mode in dict.fromkeys(a.modes):
            folder=out/mode;folder.mkdir(exist_ok=True);record=folder/'report.json'
            if record.exists():
                saved=json.loads(record.read_text())
                if digest(folder/'library.fasta')!=saved['fasta_sha256']:raise ValueError('Changed saved output')
                results[mode]=saved;continue
            # Resume from transactional, hash-checked accepted indices. The
            # continued local search uses deterministic fresh epoch proposals.
            start_ids=ids;checkpoint_path=folder/'checkpoint.npz';old_trace=[]
            if checkpoint_path.exists():
                q=np.load(checkpoint_path,allow_pickle=False);start_ids=q['ids'];old_trace=json.loads(str(q['trace']))
                if len(start_ids)!=len(ids) or len(set(map(int,start_ids)))!=len(ids) or np.any(start_ids<0) or np.any(start_ids>=len(seqs)):raise ValueError('Invalid checkpoint')
                if str(q['hash'])!=fingerprint(start_ids.tolist()):raise ValueError('Checkpoint hash mismatch')
            def checkpoint(selected,trace):
                temp=folder/'checkpoint.tmp.npz';np.savez_compressed(temp,ids=selected,trace=np.array(json.dumps(old_trace+trace)),hash=np.array(fingerprint(selected.tolist())));temp.replace(checkpoint_path)
            remaining=max(0,a.epochs-len(old_trace))
            final,search=refine(seqs,start_ids,e,de,props,rp,values,similarity,cover.bins,cover.weights,limits,a.potency_floor,mode,remaining,a.workers,a.seed+len(old_trace),checkpoint,original_guard)
            search['trace']=old_trace+search['trace'];m=search['final'];chosen=sorted(seqs[i] for i in final);write_fasta(folder/'library.fasta',chosen);np.save(folder/'selected_indices.npy',final)
            neighbors=neighbor_stats(chosen,a.workers);conformity=seqme_properties(chosen,dev);failed=[]
            if m['fbd']>limits['fbd']:failed.append('embedding_fbd')
            if m['region_l1']>limits['region_l1_max']:failed.append('regional_coverage')
            for k in NAMES:
                if m['wasserstein'][k]>limits['wasserstein'][k]:failed.append(k+'_wasserstein')
            if conformity['conformity_mean']<limits['conformity_min']:failed.append('conformity')
            for k in ['near_neighbor_fraction_gt_0_8','mean_neighbors_gt_0_8']:
                if neighbors[k]>min(baseline['neighbors'][k],source_neighbors[k])+1e-9:failed.append(k)
            if neighbors['nearest_distance_q10']<source_neighbors['nearest_distance_q10']-.01:failed.append('nearest_distance_q10')
            if m['known_mean']>baseline['known_mean']+1e-6 or m['known_p90']>baseline['known_p90']+.01 or m['known_fraction_le_0_8']<baseline['known_fraction_le_0_8']-.005:failed.append('novelty')
            for k in ['predicted_mic_um_mean','predicted_mic_um_median','predicted_mic_um_p90']:
                if m['activity'][k]>source_report['metrics']['activity'][k]*1.03+1e-8:failed.append(k+'_source_guard')
            if m['activity']['amplify_mean']<source_report['metrics']['activity']['amplify_mean']-.01:failed.append('amplify_source_guard')
            if m['activity']['predicted_fraction_le_16uM']<a.potency_floor:failed.append('potency_floor')
            rng=np.random.default_rng(a.seed+811);sample=[chosen[i] for i in rng.choice(len(chosen),min(5000,len(chosen)),replace=False)]
            held={'properties':seqme_properties(chosen,audit),'esm2':embedding_metrics(cache.embed(sample),cache.embed(audit),de,1000,3,a.seed+811),'chemistry':chemistry_diagnostics(chosen)}
            result={'mode':mode,'search':search,'metrics':m,'neighbors':neighbors,'conformity_development':conformity,'held_out_audit':held,'failed_guards':failed,'accepted':not failed,'potency_floor':a.potency_floor,'fasta_sha256':digest(folder/'library.fasta'),'warning':'Local surrogates only; official ESM-C/alignment, safety and final submission replay unverified.'}
            json_write(record,result);results[mode]=result
        eligible=[k for k,v in results.items() if v['accepted']]
        winner=min(eligible,key=lambda k:(results[k]['metrics']['fbd'],results[k]['metrics']['region_l1'],-results[k]['metrics']['activity']['predicted_fraction_le_16uM'])) if eligible else None
        summary={'complete':True,'recommended_mode':winner,'source_policy':'hybrid','source_metrics':source_report['metrics'],'limits':limits,'results':results,'note':'No production or earlier selector outputs overwritten. A null recommendation means no result passed all guards.'}
        json_write(out/'comparison.json',summary)
        lines=['# Hybrid refinement','','| Mode | MIC<=16 | Median MIC | Full development FBD | Regional L1 | Close-neighbor fraction | Failed guards |','|---|---:|---:|---:|---:|---:|---|']
        for k,v in results.items():lines.append(f"| {k} | {v['metrics']['activity']['predicted_fraction_le_16uM']:.2%} | {v['metrics']['activity']['predicted_mic_um_median']:.3f} | {v['metrics']['fbd']:.5f} | {v['metrics']['region_l1']:.5f} | {v['neighbors']['near_neighbor_fraction_gt_0_8']:.2%} | {', '.join(v['failed_guards'])} |")
        lines+=['',f'Recommended mode: {winner or "none"}.',summary['note']];(out/'comparison.md').write_text('\n'.join(lines)+'\n');print('HYBRID REFINEMENT COMPLETE',out,'recommended:',winner,flush=True)
    finally:cache.close()

if __name__=='__main__':main()
