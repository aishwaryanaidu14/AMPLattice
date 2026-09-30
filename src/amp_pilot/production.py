"""Resumable production pipeline. Local audit surrogates are not official scores."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,shutil,subprocess,sys,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from .common import read_fasta,write_fasta,json_write,digest,properties,seed_all,AA
from .production_select import GuardedSelector,potency_utility,max_similarity,neighbor_stats,repair_neighbors
from .selector import allocate

VERSION='production-1'
COLS=['amplify_prob','mbc-attention_MIC_uM']
SCHEMA={'models':[{'column':c,'family':f,'kind':k,'target':f} for c,f,k in zip(COLS,['amplify','mbc-attention'],['amp_probability','mic_um'])]}

def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def valid(seqs,label):
    if not seqs or len(set(seqs))!=len(seqs) or any(not 8<=len(s)<=50 or not set(s)<=set(AA) for s in seqs):raise ValueError('Invalid/duplicate '+label)

def csv_rows(path):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if not {'sequence',*COLS}<=set(reader.fieldnames or []):raise ValueError('Score columns missing: '+str(path))
        result={}
        for row in reader:
            seq=row['sequence'].strip().upper();v=np.array([float(row[c]) for c in COLS])
            if seq in result or not np.isfinite(v).all() or np.any(v[:-1]<0) or np.any(v[:-1]>1) or v[-1]<=0:raise ValueError('Invalid/duplicate score in '+str(path))
            result[seq]=v
    return result

def write_scores(path,seqs,table):
    temp=Path(str(path)+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sequence',*COLS])
        for s in seqs:writer.writerow([s,*[format(float(x),'.17g') for x in table[s]]])
    temp.replace(path)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',default='runs/pilot');p.add_argument('--out',default='runs/production_v1')
    p.add_argument('--battle-repo',default=str(Path.home()/'battleamp-snakemake'));p.add_argument('--snakemake')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--candidates',type=int,default=250000);p.add_argument('--library-size',type=int,default=50000);p.add_argument('--top-k',type=int,default=100)
    p.add_argument('--chunk-size',type=int,default=10000);p.add_argument('--score-batch',type=int,default=20000)
    p.add_argument('--max-extra-candidates',type=int,default=50000);p.add_argument('--rounds',type=int,default=2000)
    p.add_argument('--repair-proposals',type=int,default=3000);p.add_argument('--distribution-tolerance',type=float,default=.10)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--esm-batch',type=int,default=32)
    p.add_argument('--eval-n',type=int,default=1000);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--generic-reference');p.add_argument('--mmseqs',default=None,help='Optional MMseqs2 executable; omitted by default to skip the local alignment audit')
    p.add_argument('--esmc',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--allow-missing-audits',action=argparse.BooleanOptionalAction,default=True,help='Keep the run complete when optional audits are skipped; report remains marked incomplete')
    p.add_argument('--stage',choices=['all','generate','score','select','audit'],default='all')
    a=p.parse_args(argv)
    if min(a.library_size,a.top_k,a.chunk_size,a.score_batch,a.workers,a.rounds,a.eval_n,a.repeats,a.esm_batch)<1 or a.library_size>a.candidates or a.top_k>a.library_size or a.eval_n<4 or a.max_extra_candidates<0 or a.repair_proposals<0 or a.distribution_tolerance<0:p.error('Invalid sizes or tolerances')
    with threadpool_limits(limits=a.workers):run(a)


def run(a):
    import torch
    from .evaluate import ESMCache
    from .phase1 import selection_properties
    torch.set_num_threads(a.workers);seed_all(a.seed)
    pilot=Path(a.pilot).expanduser().resolve();root=Path(a.out).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    repo=Path(a.battle_repo).expanduser().resolve()
    snake=a.snakemake or shutil.which('snakemake')
    if a.stage in ['all','score']:
        if not snake or not Path(snake).is_file() or not (repo/'profile').is_dir():raise RuntimeError('Activate the working BATTLE environment and pass its --snakemake executable')
    mmseqs=shutil.which(a.mmseqs) if a.mmseqs else None
    if a.stage in ['all','audit'] and not mmseqs and not a.allow_missing_audits:raise RuntimeError('MMseqs2 audit requested but no --mmseqs executable was supplied.')
    if a.stage in ['all','audit'] and a.esmc and not shutil.which('uv'):raise RuntimeError('uv executable required for isolated ESM-C audit')
    manifest=json.loads((pilot/'manifest.json').read_text())
    for name,h in manifest['outputs'].items():
        if not (pilot/name).is_file() or digest(pilot/name)!=h:raise RuntimeError('Prepared pilot data missing/changed: '+name)
    for kind in ['flow','ar']:
        if not (pilot/kind/'best.pt').is_file():raise RuntimeError('Missing trained '+kind+' checkpoint')
    dev=read_fasta(pilot/'dev.fasta',False);audit=read_fasta(pilot/'audit.fasta',False)
    valid(dev,'development');valid(audit,'audit')
    if set(dev)&set(audit):raise ValueError('Reference overlap')
    known=sorted(set(read_fasta(pilot/'all_positive.fasta'))|set(read_fasta(pilot/'forbidden.fasta',False)))
    known=[s for s in known if set(s)<=set(AA) and s];known_set=set(known)
    source_hashes={'manifest':digest(pilot/'manifest.json'),**{k:digest(pilot/k/'best.pt') for k in ['flow','ar']}}
    semantic={k:getattr(a,k) for k in ['candidates','library_size','top_k','chunk_size','max_extra_candidates','rounds','repair_proposals','distribution_tolerance','seed','device']}
    signature={'version':VERSION,'source_hashes':source_hashes,'configuration':semantic}
    pin=root/'run_manifest.json'
    if pin.exists() and json.loads(pin.read_text())['signature']!=signature:raise RuntimeError('Production configuration/checkpoints changed. Use a new --out directory to preserve this run.')
    if not pin.exists():json_write(pin,{'signature':signature,'proposal_mix_new_draws':{'heun_32':.7,'blend35_heun_32':.2,'ar_temp_1.1':.1},'note':'Engineering allocation, not a tuned optimum. Sampling streams are independent of solver comparisons.'})
    # Isolated sampling workspace: preserve the original checkpoint and pilot candidates.
    work=root/'sampling';work.mkdir(exist_ok=True)
    for name in ['train.fasta','all_positive.fasta','forbidden.fasta']:
        dest=work/name
        if not dest.exists():dest.symlink_to(pilot/name)
    for kind in ['flow','ar']:
        (work/kind).mkdir(exist_ok=True);dest=work/kind/'best.pt'
        if not dest.exists():dest.symlink_to(pilot/kind/'best.pt')
    config=SimpleNamespace(device=a.device,esm_model='esm2_t6_8M_UR50D',esm_batch=a.esm_batch)
    cache=ESMCache(pilot,config)
    try:
        if a.stage in ['all','generate']:
            seqs=build_pool(a,pilot,root,work,known_set)
            seqs=replenish(a,root,work,seqs,dev,known_set,cache)
            if a.stage=='generate':print('GENERATION COMPLETE',root/'candidates.fasta');return
        else:
            seqs=read_fasta(root/'candidates.fasta',False);valid(seqs,'candidate pool')
        valid(seqs,'candidate pool')
        if set(seqs)&known_set:raise RuntimeError('Candidate pool includes excluded known sequences')
        if len(seqs)<a.candidates:raise RuntimeError('Candidate pool is not complete')
        if a.stage in ['all','score']:
            score_candidates(a,pilot,root,repo,snake,seqs)
            if a.stage=='score':print('SCORING COMPLETE',root/'scores.csv');return
        table=csv_rows(root/'scores.csv')
        if set(seqs)!=set(table):raise RuntimeError('Scoring pool mismatch')
        values=np.stack([table[s] for s in seqs]);de=cache.embed(dev);props_ref=selection_properties(dev)
        stamp=root/'selection_complete.json';selection_config={'pool':digest(root/'candidates.fasta'),'scores':digest(root/'scores.csv'),'signature':signature,'code':{name:digest(Path(__file__).with_name(name)) for name in ['production.py','production_select.py','selector.py','phase1.py','evaluate.py','common.py']}}
        reusable=False
        if stamp.exists():
            saved=json.loads(stamp.read_text());reusable=saved['config']==selection_config and all((root/k).exists() and digest(root/k)==v for k,v in saved['files'].items())
        if a.stage in ['all','select'] and not reusable:
            e=cache.embed(seqs);props=selection_properties(seqs)
            known_selection=sorted(known_set-set(audit))
            sim_path=root/'candidate_known_similarity.npz';sim_signature=fingerprint({'pool':digest(root/'candidates.fasta'),'reference':known_selection})
            if sim_path.exists():
                saved=np.load(sim_path,allow_pickle=False)
                similarity=saved['similarity'] if str(saved['signature'])==sim_signature else None
            else:similarity=None
            if similarity is None:
                similarity=max_similarity(seqs,known_selection,a.workers)
                np.savez_compressed(sim_path,signature=np.array(sim_signature),similarity=similarity)
            selector=GuardedSelector(e,de,props,props_ref,a.seed)
            baseline,ids,selection,snapshots=selector.optimize(seqs,values,similarity,a.library_size,a.rounds,a.distribution_tolerance)
            repaired,repair=repair_neighbors(selector,seqs,ids,baseline,values,similarity,selection,a.workers,a.repair_proposals)
            # Exact development checks choose among saved feasible trajectory points; no audit optimization.
            ids,checks=choose_feasible(a,seqs,e,de,props,props_ref,dev,values,similarity,baseline,[repaired,*snapshots[::-1]])
            selected=[seqs[i] for i in sorted(ids,key=lambda j:seqs[j])]
            baseline_seqs=[seqs[i] for i in sorted(baseline,key=lambda j:seqs[j])]
            write_fasta(root/'baseline.fasta',baseline_seqs);write_fasta(root/'library.fasta',selected)
            full_known=max_similarity(selected,known,a.workers)
            top=top_candidates(selected,table,full_known,a.top_k)
            write_fasta(root/'top.fasta',top);np.save(root/'selected_known_similarity.npy',full_known)
            selection.update(repair=repair,exact_development_checks=checks,selected_count=len(selected),selection_seed=a.seed)
            json_write(root/'selection.json',selection)
            files=['library.fasta','baseline.fasta','top.fasta','selection.json','selected_known_similarity.npy']
            json_write(stamp,{'config':selection_config,'files':{k:digest(root/k) for k in files}})
        elif a.stage in ['all','select']:print('Reuse validated production selection',flush=True)
        elif not reusable:raise RuntimeError('Selection is absent/stale; run --stage select before audit')
        if a.stage=='select':print('SELECTION COMPLETE',root/'library.fasta');return
        audit_results(a,pilot,root,seqs,dev,audit,known,table,cache,mmseqs)
    finally:cache.close()


def build_pool(a,pilot,root,work,known):
    from .generate import generate
    completion=root/'pool_base_complete.json';output=root/'candidates.fasta'
    if completion.exists() and output.exists():
        record=json.loads(completion.read_text())
        # Subsequent replenishment changes candidates.fasta; its own completion is validated separately.
        if (root/'replenishment_complete.json').exists():return read_fasta(output,False)
        if record['sha256']==digest(output):return read_fasta(output,False)
    pool=set();sources={}
    for kind in ['flow','ar']:
        for path in sorted((pilot/kind).glob('*/eligible.fasta')):
            meta_path=path.with_name('generation.json')
            if not meta_path.exists():continue
            meta=json.loads(meta_path.read_text())
            if meta.get('settings',{}).get('checkpoint_sha256')!=digest(pilot/kind/'best.pt'):continue
            if meta.get('files',{}).get('eligible.fasta')!=digest(path):raise RuntimeError('Changed pilot candidate cache: '+str(path))
            values=read_fasta(path,False)
            if values:valid(values,str(path))
            pool.update(s for s in values if s not in known)
            sources[str(path.relative_to(pilot))]={'hash':digest(path),'n':len(values)}
    stream=0
    while len(pool)<a.candidates:
        slot=stream%10;kind='ar' if slot==9 else 'flow'
        variant=f'stream{stream:04d}_temp_1.1' if kind=='ar' else (f'blend35_stream{stream:04d}_heun_32' if slot>=7 else f'stream{stream:04d}_heun_32')
        args=SimpleNamespace(out=str(work),samples=a.chunk_size,sample_batch=128,seed=a.seed+100000+stream,device=a.device)
        dest=generate(args,kind,variant);new=read_fasta(dest/'eligible.fasta',False);before=len(pool);pool.update(s for s in new if s not in known)
        print('Candidate pool',len(pool),'unique; stream',stream,'added',len(pool)-before,flush=True)
        stream+=1
        if stream>max(100,20*a.candidates//a.chunk_size):raise RuntimeError('Insufficient unique yield; inspect sampling diagnostics')
    seqs=sorted(pool);write_fasta(output,seqs)
    json_write(completion,{'sha256':digest(output),'count':len(seqs),'new_streams':stream,'prior_pools':sources})
    return seqs


def replenish(a,root,work,seqs,dev,known,cache):
    from .phase1 import selection_properties
    from .generate import generate
    record=root/'replenishment_complete.json';output=root/'candidates.fasta'
    if record.exists():
        saved=json.loads(record.read_text())
        if saved['sha256']==digest(output):return seqs
        raise RuntimeError('Production candidate FASTA changed after replenishment')
    de=cache.embed(dev);rp=selection_properties(dev);pool=set(seqs);trace=[];extra=0
    for cycle in range(3):
        seqs=sorted(pool);sel=GuardedSelector(cache.embed(seqs),de,selection_properties(seqs),rp,a.seed)
        desired=allocate(a.library_size,sel.cover.weights,np.full(len(sel.cover.weights),a.library_size))
        short=np.maximum(0,desired-sel.cover.capacity)
        trace.append({'cycle':cycle,'candidate_count':len(seqs),'target_shortfall':int(short.sum()),'capacity':sel.cover.capacity.tolist(),'desired':desired.tolist()})
        if not short.any() or extra>=a.max_extra_candidates or cycle==2:break
        labels=sel.cover.km.labels_;targets=[]
        # Replicate reference templates proportional to the deficient regions' demand.
        for j in np.flatnonzero(short):
            refs=[dev[i] for i in np.flatnonzero(labels==j)]
            if refs:targets.extend((refs*((int(short[j])+len(refs)-1)//len(refs)))[:int(short[j])])
        condition=root/f'replenish_{cycle}.fasta';write_fasta(condition,targets)
        draws=min(a.max_extra_candidates-extra,max(a.chunk_size,4*int(short.sum())))
        args=SimpleNamespace(out=str(work),samples=draws,sample_batch=128,seed=a.seed+900000+cycle,device=a.device,condition_fasta=str(condition))
        dest=generate(args,'flow',f'replenish{cycle}_heun_32');pool.update(s for s in read_fasta(dest/'eligible.fasta',False) if s not in known);extra+=draws
    seqs=sorted(pool);write_fasta(output,seqs)
    json_write(record,{'sha256':digest(output),'count':len(seqs),'extra_raw_draws':extra,'trace':trace,'remaining_shortfall':trace[-1]['target_shortfall'],'note':'Additional conditioning follows development-region shortages; capped quotas remain if supply is insufficient.'})
    return seqs


def git_info(repo):
    try:
        result=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10,check=True)
        status=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,timeout=10,check=True)
        return {'head':result.stdout.strip(),'dirty':bool(status.stdout.strip())}
    except (OSError,subprocess.SubprocessError):return {'head':None,'dirty':None}


def score_candidates(a,pilot,root,repo,snake,seqs):
    table={};orig=pilot/'experiments_v3/battle_scores.csv'
    if orig.exists():table.update(csv_rows(orig))
    if (root/'scores.csv').exists():table.update(csv_rows(root/'scores.csv'))
    missing=[s for s in seqs if s not in table];folder=root/'scoring';folder.mkdir(exist_ok=True)
    env=os.environ.copy();env['PATH']=str(Path(snake).resolve().parent)+os.pathsep+env.get('PATH','')
    helper=Path(__file__).with_name('run_battle.py')
    for start in range(0,len(missing),a.score_batch):
        chunk=missing[start:start+a.score_batch];key=fingerprint(chunk)[:20];fasta=folder/(key+'.fasta');output=folder/(key+'.csv');stamp=folder/(key+'.complete.json')
        write_fasta(fasta,chunk);reuse=False
        if output.exists() and stamp.exists():
            m=json.loads(stamp.read_text());reuse=m.get('input_hash')==digest(fasta) and m.get('output_hash')==digest(output) and output.with_suffix('.report.json').exists() and m.get('report_hash')==digest(output.with_suffix('.report.json'))
        if not reuse:
            subprocess.run([sys.executable,str(helper),'--repo',str(repo),'--fasta',str(fasta),'--output',str(output),'--cores',str(a.workers),'--models','amplify,mbc-attention'],env=env,check=True)
            checked=csv_rows(output)
            if set(checked)!=set(chunk):raise RuntimeError('BATTLE batch returned wrong sequence set')
            json_write(stamp,{'input_hash':digest(fasta),'output_hash':digest(output),'report_hash':digest(output.with_suffix('.report.json'))})
        table.update(csv_rows(output));print('BATTLE batches',min(start+len(chunk),len(missing)),'/',len(missing),flush=True)
    if not all(s in table for s in seqs):raise RuntimeError('Incomplete predictions')
    write_scores(root/'scores.csv',seqs,table);json_write(root/'scores.schema.json',SCHEMA)
    json_write(root/'scores_provenance.json',{'pool_hash':digest(root/'candidates.fasta'),'score_hash':digest(root/'scores.csv'),'reused_pilot_csv':str(orig) if orig.exists() else None,'reused_pilot_hash':digest(orig) if orig.exists() else None,'battle_git_head':git_info(repo),'device_policy':'AMPlify/MBC preserve CUDA visibility; AMPeppy omitted','battle_repo':str(repo),'batch_count':len(list(folder.glob('*.complete.json'))),'warning':'External model weights/environments must be pinned when preparing the final submission.'})


def activity_summary(values):
    v=np.asarray(values,float);mic=v[:,-1]
    return {'n':len(v),'amplify_mean':float(v[:,0].mean()),
        'predicted_mic_um_mean':float(mic.mean()),'predicted_mic_um_median':float(np.median(mic)),
        'predicted_mic_um_p90':float(np.quantile(mic,.9)),'predicted_mic_um_geometric_mean':float(np.exp(np.log(mic).mean())),
        'predicted_fraction_le_16uM':float(np.mean(mic<=16))}


def choose_feasible(a,seqs,e,de,props,reference_props,dev,values,similarity,baseline,options):
    """Development checks only. Reference-balanced baseline is the explicit fallback."""
    from scipy.stats import wasserstein_distance
    from .evaluate import fbd
    from .phase1 import seqme_properties
    names=['length','charge','hydrophobicity','hydrophobic_moment']
    def cheap(ids):
        return {'esm2_full_fbd':fbd(e[ids],de),
            'wasserstein_std_units':{name:float(wasserstein_distance(props[ids,j],reference_props[:,j])/max(reference_props[:,j].std(),.01)) for j,name in enumerate(names)},
            'activity':activity_summary(values[ids]),'known_similarity_mean':float(similarity[ids].mean()),
            'known_similarity_p90':float(np.quantile(similarity[ids],.9)),
            'known_fraction_le_0_8':float(np.mean(similarity[ids]<=.8))}
    b=cheap(baseline);b['seqme']=seqme_properties([seqs[i] for i in baseline],dev)
    b['neighbors']=neighbor_stats([seqs[i] for i in baseline],a.workers)
    attempts=[];seen=set();tol=a.distribution_tolerance
    for ix,ids in enumerate(options):
        key=fingerprint(sorted(map(int,ids)))
        if key in seen:continue
        seen.add(key)
        if set(ids)==set(baseline):continue
        m=cheap(ids);failed=[]
        if m['esm2_full_fbd']>max(1e-6,b['esm2_full_fbd']*(1+tol)):failed.append('esm2_full_fbd')
        for k in names:
            if m['wasserstein_std_units'][k]>max(1e-6,b['wasserstein_std_units'][k]*(1+tol)):failed.append(k+'_wasserstein')
        for k in ['predicted_mic_um_mean','predicted_mic_um_median','predicted_mic_um_p90']:
            if m['activity'][k]>b['activity'][k]+1e-8:failed.append(k)
        for k in ['amplify_mean']:
            if m['activity'][k]<b['activity'][k]-.01-1e-8:failed.append(k)
        if m['known_similarity_mean']>b['known_similarity_mean']+1e-8:failed.append('known_similarity_mean')
        if m['known_similarity_p90']>b['known_similarity_p90']+.01:failed.append('known_similarity_p90')
        if m['known_fraction_le_0_8']<b['known_fraction_le_0_8']-.005:failed.append('known_novel_fraction')
        if not failed:
            m['seqme']=seqme_properties([seqs[i] for i in ids],dev)
            if m['seqme']['conformity_mean']<b['seqme']['conformity_mean']-.02:failed.append('conformity_mean')
        if not failed:
            m['neighbors']=neighbor_stats([seqs[i] for i in ids],a.workers)
            for k in ['near_neighbor_fraction_gt_0_8','mean_neighbors_gt_0_8']:
                if m['neighbors'][k]>b['neighbors'][k]+1e-10:failed.append(k)
            if m['neighbors']['nearest_distance_q10']<b['neighbors']['nearest_distance_q10']-.01:failed.append('nearest_distance_q10')
        attempts.append({'trajectory_option':ix,'metrics':m,'failed_guards':failed})
        print('Development acceptance',ix,'failed:',failed,flush=True)
        if not failed:
            return ids,{'baseline':b,'selected':m,'attempts':attempts,'used_baseline_fallback':False,
                'policy':'Try repaired solution first, then earlier checkpoints; no audit-set feedback.'}
    print('No optimized checkpoint passed all guards; preserving reference-balanced baseline.',flush=True)
    return baseline.copy(),{'baseline':b,'selected':b,'attempts':attempts,'used_baseline_fallback':True,
        'warning':'No potency improvement was accepted. Read this result before treating the output as a production improvement.'}


def top_candidates(seqs,table,known_similarity,k):
    from rapidfuzz import fuzz
    values=np.stack([table[s] for s in seqs]);utility=potency_utility(values,known_similarity)
    eligible=[i for i in range(len(seqs)) if known_similarity[i]<=.8]
    order=sorted(eligible,key=lambda i:(-utility[i],values[i,-1],seqs[i]));chosen=[]
    for i in order:
        if all(fuzz.ratio(seqs[i],s)<=80 for s in chosen):chosen.append(seqs[i])
        if len(chosen)==k:return chosen
    raise RuntimeError(f'Only {len(chosen)} of {k} top candidates meet known-reference and mutual >80% exclusion. No invalid top file was written; inspect the pool/selection.')


def audit_results(a,pilot,root,seqs,dev,audit,known,table,cache,mmseqs):
    import gc,platform
    from importlib.metadata import version,PackageNotFoundError
    import torch
    from .phase1 import (seqme_properties,selection_properties,embedding_metrics,reference_floor,
                         sequence_metrics,chemistry_diagnostics,load_extra_embeddings)
    from .alignment import alignment_novelty
    from .evaluate import fbd
    selection=json.loads((root/'selection.json').read_text());checks=selection['exact_development_checks']
    libraries={name:read_fasta(root/(name+'.fasta'),False) for name in ['library','baseline','top']}
    if len(libraries['library'])!=a.library_size or len(libraries['top'])!=a.top_k:raise RuntimeError('Wrong final sizes')
    for label,ss in libraries.items():
        valid(ss,label)
        if set(ss)&set(known):raise RuntimeError('Known-sequence exclusion failed: '+label)
    if not set(libraries['top'])<=set(libraries['library']):raise RuntimeError('Top set is not a subset')
    top_sim=max_similarity(libraries['top'],known,a.workers)
    top_neighbors=neighbor_stats(libraries['top'],a.workers)
    if np.any(top_sim>.8) or top_neighbors['undirected_edges_gt_0_8']:raise RuntimeError('Top-set >80% similarity validation failed')
    incumbent=pilot/'phase1/flow/euler_16/novelty/library.fasta'
    incumbent_note='Original 1,250-sequence incumbent is absent or its hash differs; not compared.'
    if incumbent.exists() and digest(incumbent)=='42f942e2e9a9b89f11fb11eb08f480f74933fb9a7595bd1c43945dd5a962a29d':
        ss=read_fasta(incumbent,False)
        if all(s in table for s in ss):libraries['pilot_incumbent_1250']=ss;incumbent_note='Included only as a small-set comparator, not a 50k alternative.'
    if a.generic_reference:
        generic=read_fasta(Path(a.generic_reference).expanduser(),False);valid(generic,'generic reference')
        generic_label='User-supplied generic reference; verify provenance against organizer requirements.'
    else:
        generic=sorted(set(s for path in [pilot/(part+'_negative.fasta') for part in ['train','dev','audit']] if path.exists() for s in read_fasta(path))-set(known))
        generic_label='Labelled-negative proxy, NOT an official generic-peptide reference.'
    audit_dir=root/'audit';audit_dir.mkdir(exist_ok=True)
    report={'complete':False,'configuration':vars(a),'selection':selection,'predictor_models':['amplify','mbc-attention'],'omitted_predictors':['ampeppy'],
        'scope':'Local Phase 1 surrogates. Predictions are not measured MIC or official hidden scores.',
        'reference_hashes':{'development':digest(pilot/'dev.fasta'),'audit':digest(pilot/'audit.fasta'),'known':fingerprint(known),'generic':fingerprint(generic)},
        'generic_reference':{'label':generic_label,'n':len(generic)},'incumbent_note':incumbent_note,
        'candidate_count':len(seqs),'libraries':{},'errors':[],
        'unavailable_official_details':['Hidden score weights and exact evaluation configuration','Organizer generic reference if not provided','Validated synthesizability model/thresholds and experimental activity','Official top-100 MarLys/MMseqs2 identity gate: local Indel similarity is not equivalent'],
        'conformity_explanation':'seqme charge/hydrophobic-moment density conformity under reference KDE defaults; conformity alone does not establish matching distributions.',
        'sampling_note':'Activity/properties and final Indel neighbors use all selected sequences. Embedding distribution metrics and MMseqs alignment use explicit fixed-size samples.'}
    json_write(root/'report.json',report)
    # Audit products are cached separately; changing references/settings invalidates them.
    audit_signature=fingerprint({'selection':digest(root/'selection_complete.json'),'refs':report['reference_hashes'],
        'eval_n':a.eval_n,'repeats':a.repeats,'seed':a.seed,'runner':digest(__file__),'exporter':digest(Path(__file__).resolve().parents[1]/'export_esmc.py'),'alignment_code':digest(Path(__file__).with_name('alignment.py')),'mmseqs':mmseqs})
    def cached(name,fn):
        path=audit_dir/(name+'.json')
        if path.exists():
            obj=json.loads(path.read_text())
            if obj.get('signature')==audit_signature:return obj['result']
        result=fn();json_write(path,{'signature':audit_signature,'result':result});return result
    # Fix a sequence reservoir independently of scores. ESM2 and ESM-C use identical strings.
    reservoirs={}
    for name,ss in libraries.items():
        count=min(len(ss),max(5000,a.eval_n));ids=np.random.default_rng(a.seed+811).choice(len(ss),count,replace=False)
        reservoirs[name]=[ss[i] for i in ids]
    references={'amp_audit':audit}
    if len(generic)>=4:references['generic_proxy' if not a.generic_reference else 'generic']=generic
    de=cache.embed(dev);reference_embeddings={k:cache.embed(v) for k,v in references.items()}
    report['esm2_reference_floor']=cached('esm2_reference_floor',lambda:reference_floor(reference_embeddings['amp_audit'],a.eval_n,a.repeats,a.seed+990))
    for name,ss in libraries.items():
        print('Audit',name,len(ss),flush=True)
        def core():
            out={'sha256':digest(root/(name+'.fasta')) if name in ['library','baseline','top'] else digest(incumbent),
                 'activity':activity_summary(np.stack([table[s] for s in ss])),
                 'seqme_properties':seqme_properties(ss,audit),'chemistry':chemistry_diagnostics(ss),
                 'sequence_sample':sequence_metrics(ss,known,a.eval_n,a.seed+811,a.workers)}
            if name in ['library','baseline']:out['full_selected_neighbors']=checks['selected' if name=='library' else 'baseline']['neighbors']
            elif name=='top':out['full_selected_neighbors']=top_neighbors
            else:out['full_selected_neighbors']=neighbor_stats(ss,a.workers)
            if name=='library':sim=np.load(root/'selected_known_similarity.npy',allow_pickle=False)
            elif name=='top':sim=top_sim
            else:sim=max_similarity(ss,known,a.workers)
            out['full_known_similarity']={'reference_n':len(known),'n':len(ss),'mean':float(sim.mean()),'median':float(np.median(sim)),'p90':float(np.quantile(sim,.9)),'fraction_le_0_8':float(np.mean(sim<=.8))}
            emb=cache.embed(reservoirs[name]);out['esm2']={label:embedding_metrics(emb,ref,de,a.eval_n,a.repeats,a.seed+811) for label,ref in reference_embeddings.items()}
            return out
        report['libraries'][name]=cached(name+'_core',core)
        json_write(root/'report.json',report)
    if mmseqs:
        for name,ss in reservoirs.items():
            try:report['libraries'][name]['alignment']=cached(name+'_alignment',lambda ss=ss,name=name:alignment_novelty(ss[:a.eval_n],known,audit_dir/('mmseqs_'+name),mmseqs,a.workers))
            except Exception as ex:report['errors'].append({'stage':'mmseqs','library':name,'error':str(ex)})
            json_write(root/'report.json',report)
    else:report['errors'].append({'stage':'mmseqs','error':'Executable unavailable; explicitly permitted by --allow-missing-audits.'})
    if a.esmc:
        try:
            cache.model=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            union=sorted(set(dev).union(*references.values(),*reservoirs.values()))
            input_path=audit_dir/'esmc_input.fasta';output=audit_dir/'esmc_embeddings.npz';write_fasta(input_path,union)
            script=Path(__file__).resolve().parents[1]/'export_esmc.py'
            subprocess.run(['uv','run','--no-project','--python','3.11','--script',str(script),'--fasta',str(input_path),'--output',str(output),'--device',a.device,'--batch','16'],check=True)
            all_e,model=load_extra_embeddings(output,union)
            if not model.startswith('esmc_300m:'):raise RuntimeError('Unexpected ESM-C model provenance')
            index={s:i for i,s in enumerate(union)}
            get=lambda ss:all_e[[index[s] for s in ss]]
            ed=get(dev);refs={k:get(v) for k,v in references.items()}
            report['esmc_model']=model
            report['esmc_reference_floor']=cached('esmc_reference_floor',lambda:reference_floor(refs['amp_audit'],a.eval_n,a.repeats,a.seed+990))
            for name,ss in reservoirs.items():
                report['libraries'][name]['esmc']=cached(name+'_esmc',lambda ss=ss:{label:embedding_metrics(get(ss),ref,ed,a.eval_n,a.repeats,a.seed+811) for label,ref in refs.items()})
        except Exception as ex:report['errors'].append({'stage':'esmc','error':str(ex)})
    else:report['errors'].append({'stage':'esmc','error':'Explicitly disabled with --no-esmc.'})
    versions={}
    for pkg in ['torch','numpy','scipy','scikit-learn','rapidfuzz','fair-esm','seqme']:
        try:versions[pkg]=version(pkg)
        except PackageNotFoundError:versions[pkg]='unavailable'
    code_root=Path(__file__).resolve().parents[1]
    provenance={'python':sys.version,'platform':platform.platform(),'packages':versions,
        'code_sha256':{str(p.relative_to(code_root)):digest(p) for p in sorted(code_root.rglob('*.py')) if not any(x in p.parts for x in ['.venv','__pycache__'])},
        'lock_sha256':digest(code_root/'uv.lock'),'run_manifest':json.loads((root/'run_manifest.json').read_text()),
        'outputs':{name:digest(root/name) for name in ['candidates.fasta','scores.csv','library.fasta','top.fasta','selection.json']},
        'single_command_current':'uv run --project amp-phase1-upgrade --locked amp-production [documented arguments]',
        'final_uv_run_generate_ready':False,
        'remaining_submission_work':['Resolve and run organizer top-100 identity check against the required MarLys reference; Indel is not MMseqs2 alignment identity','Package trained checkpoints and input data with stable paths','Pin BATTLE repository, both selected predictor weights and environments, ESM2 and ESM-C weights','Implement uv run generate writing generate/library.fasta and generate/top.fasta','Verify two fresh runs produce identical hashes in the target environment; cache replay alone is not that test']}
    json_write(root/'provenance.json',provenance)
    write_scores(root/'library_scores.csv',libraries['library'],table);write_scores(root/'top_scores.csv',libraries['top'],table)
    report['complete']=not report['errors'];report['core_complete']=True
    report['status']='complete_local_audit' if report['complete'] else 'incomplete_optional_audits'
    report['decision_note']='Baseline fallback: no guarded potency improvement accepted.' if checks['used_baseline_fallback'] else 'Guarded selection accepted on development data; inspect the independent audit before final submission.'
    json_write(root/'report.json',report)
    lines=['# Production Phase 1 audit','',report['decision_note'],'',report['scope'],'',
           '| Set | N | Predicted MIC median (uM) | MIC <=16 | AMPlify | Conformity | Near-neighbor fraction |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['libraries'].items():
        m=r['activity'];lines.append(f"| {name} | {m['n']} | {m['predicted_mic_um_median']:.4g} | {m['predicted_fraction_le_16uM']:.2%} | {m['amplify_mean']:.4f} | {r['seqme_properties']['conformity_mean']:.4f} | {r['full_selected_neighbors']['near_neighbor_fraction_gt_0_8']:.2%} |")
    lines+=['','Full distribution, novelty, alignment, sample sizes, missing audits, and provenance: `report.json` and `provenance.json`.','',generic_label,'','Audit errors: '+json.dumps(report['errors']),
            '', 'This package is not yet a verified final `uv run generate` submission. See `provenance.json` for remaining packaging and replay checks.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('PRODUCTION OUTPUTS',root,'STATUS',report['status'],flush=True)
    if report['errors'] and not a.allow_missing_audits and any(e['stage']!='esmc' or a.esmc for e in report['errors']):
        raise RuntimeError('Audit incomplete; outputs are preserved. Inspect report.json, fix the reported dependency, and rerun with --stage audit.')


if __name__=='__main__':main()
