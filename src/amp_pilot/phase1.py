"""Phase-1-oriented selector/evaluator. Uses existing generators and cached ESM2."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from rapidfuzz import process, fuzz
from threadpoolctl import threadpool_limits
from .common import AA, read_fasta, write_fasta, properties, features, json_write, digest, seed_all
from .evaluate import ESMCache, train_proxy, select, fbd, precision_recall, diversity
from .selector import CoverageSelector, ranks, rff_fit


def validate(seqs,label):
    if not seqs or any(not 8<=len(s)<=50 or not set(s)<=set(AA) for s in seqs):
        raise ValueError(f'{label}: require nonempty canonical 8-50 residue FASTA')
    if len(set(seqs))!=len(seqs): raise ValueError(f'{label}: duplicate sequences')


def external_scores(path,schema_path,seqs):
    schema=json.loads(Path(schema_path).read_text()); models=schema['models']
    if not models:raise ValueError('Empty score schema')
    columns=[m['column'] for m in models]
    if len(set(columns))!=len(columns):raise ValueError('Duplicate score columns')
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if not {'sequence',*columns}<=set(reader.fieldnames or []):raise ValueError('Missing score columns')
        table={}
        for row in reader:
            s=row['sequence'].strip().upper()
            if s in table:raise ValueError('Duplicate sequence rows in external scores')
            table[s]=row
    family={}; summaries={}; missing=[]
    for model in models:
        col=model['column'];kind=model['kind']; fam=model['family']
        if kind not in ['amp_probability','mic_um']:raise ValueError('Score kind must be amp_probability or mic_um')
        vals=[]
        for s in seqs:
            cell=table.get(s,{}).get(col,'').strip()
            try:v=float(cell)
            except ValueError:v=float('nan')
            if np.isfinite(v) and ((kind=='amp_probability' and not 0<=v<=1) or (kind=='mic_um' and v<=0)):
                raise ValueError(f'Invalid {col}; MIC must be positive unlogged micromolar')
            vals.append(v)
        vals=np.asarray(vals); good=np.isfinite(vals)
        summary={'family':fam,'kind':kind,'target':model.get('target'),'coverage':float(good.mean()),'n_scored':int(good.sum()),'n_total':len(seqs)}
        if good.any():
            summary.update(mean=float(vals[good].mean()),median=float(np.median(vals[good])))
            if kind=='mic_um':summary['predicted_fraction_le_16uM']=float((vals[good]<=16).mean())
        summaries[col]=summary
        if not good.all():missing.append(col)
        else:family.setdefault(fam,[]).append(ranks(vals if kind=='amp_probability' else -np.log2(vals)))
    if missing:
        raise ValueError('Incomplete requested predictor coverage; do not rank partially scored libraries. Missing columns: '+', '.join(missing)+'. Choose models covering 8-50, fix failed inference, or revise schema explicitly.')
    # Target variants within a family do not get independent model votes.
    votes=np.stack([np.mean(v,axis=0) for v in family.values()])
    utility=np.clip(votes.mean(0)-.25*votes.std(0),0,1)
    return utility,{'models':summaries,'family_count':len(family),'families':list(family),
        'aggregation':'within-family percentile mean; equal family mean minus 0.25 family disagreement',
        'warning':'Percentile utility is a selection heuristic, not calibrated potency or official weights.'},table,models


def exact_mmd(x,y,bandwidth):
    if min(len(x),len(y))<2:return None
    xx=np.exp(-cdist(x,x,'sqeuclidean')/(2*bandwidth)); yy=np.exp(-cdist(y,y,'sqeuclidean')/(2*bandwidth))
    xy=np.exp(-cdist(x,y,'sqeuclidean')/(2*bandwidth))
    return float((xx.sum()-np.trace(xx))/(len(x)*(len(x)-1))+(yy.sum()-np.trace(yy))/(len(y)*(len(y)-1))-2*xy.mean())


def embedding_metrics(x,ref,development,n,repeats,seed):
    n=min(n,len(x),len(ref)); results=[]
    if n<4:return {'insufficient_sample':True}
    sub=development[:min(512,len(development))]
    dist=cdist(sub,sub,'sqeuclidean');pos=dist[dist>0];bw=max(float(np.median(pos)) if len(pos) else 1.,1e-8)
    for j in range(repeats):
        rng=np.random.default_rng(seed+j)
        a=x[rng.choice(len(x),n,replace=False)]; b=ref[np.random.default_rng(seed+j).choice(len(ref),n,replace=False)]
        m={'fbd':fbd(a,b),'mmd_unbiased':exact_mmd(a,b,bw),**precision_recall(a,b)}
        results.append(m)
    return {'sample_size':n,'repeats':repeats,'kernel_bandwidth_sq':bw,
        'mean':{k:float(np.mean([v[k] for v in results])) for k in results[0]},
        'subsample_sd':{k:float(np.std([v[k] for v in results])) for k in results[0]},
        'replicates':results,'warning':'Subsample variability, not independent-run confidence intervals. Unbiased MMD may be negative.'}


def reference_floor(ref,n,repeats,seed):
    n=min(n,len(ref)//2)
    if n<4:return {'insufficient_sample':True}
    vals=[]
    for j in range(repeats):
        ids=np.random.default_rng(seed+j).permutation(len(ref));a=ref[ids[:n]];b=ref[ids[n:2*n]]
        vals.append({'fbd':fbd(a,b),**precision_recall(a,b)})
    return {'sample_size':n,'mean':{k:float(np.mean([v[k] for v in vals])) for k in vals[0]},
            'warning':'Disjoint within-reference resampling diagnostic; sample size may differ from candidate evaluation.'}


def sequence_metrics(seqs,known,n,seed,workers):
    x=[seqs[i] for i in np.random.default_rng(seed).choice(len(seqs),min(n,len(seqs)),replace=False)]
    maxima=[]; set_known=set(known)
    for start in range(0,len(x),64):
        maxima.extend((process.cdist(x[start:start+64],known,scorer=fuzz.ratio,workers=workers,dtype=np.float32).max(1)/100).tolist())
    # Sampled greedy sequence clusters: diagnostic, not MMseqs2 cluster coverage.
    representatives=[]
    for s in x:
        if not representatives or max(fuzz.ratio(s,r) for r in representatives)<80:representatives.append(s)
    d=process.cdist(x,x,scorer=fuzz.ratio,workers=workers,dtype=np.float32)
    np.fill_diagonal(d,-1)
    return {'unique_fraction':len(set(seqs))/len(seqs),'exact_known_fraction':sum(s in set_known for s in seqs)/len(seqs),
        'sample_size':len(x),'pairwise_indel_diversity':diversity(x,seed),
        'known_max_indel_ratio_median':float(np.median(maxima)),
        'known_indel_ratio_le_0_8_fraction':float((np.asarray(maxima)<=.8).mean()),
        'sample_internal_near_duplicate_fraction':float((d.max(1)>80).mean()) if len(x)>1 else 0.,
        'sample_greedy_cluster_count_at_0_8':len(representatives),
        'warning':'Indel-ratio diagnostics; alignment normalized-bit-score novelty remains separate.'}


def seqme_properties(seqs,ref):
    import seqme as sm
    from importlib.metadata import version
    charge=sm.models.Charge();moment=sm.models.HydrophobicMoment()
    predictors=[charge,moment]
    arrays=[np.asarray(p(seqs)).reshape(-1) for p in predictors]
    references=[np.asarray(p(ref)).reshape(-1) for p in predictors]
    metrics=[sm.metrics.ConformityScore(reference=ref,predictors=predictors,kde_bandwidth='silverman')]
    df=sm.evaluate({'candidates':seqs},metrics)
    return {'package_version':version('seqme'),'charge_model':'seqme.models.Charge defaults',
        'amphiphilicity_model':'Eisenberg, window=11, angle=100 degrees, modality=mean',
        'conformity_mean':float(df.iloc[0,0]),'conformity_deviation':float(df.iloc[0,1]),
        'conformity_table':json.loads(df.to_json(orient='split')),
        'wasserstein_std_units':{k:float(wasserstein_distance(x,y)/max(y.std(),.01)) for k,x,y in zip(['charge','hydrophobic_moment'],arrays,references)},
        'warning':'Uses public seqme defaults, not undisclosed organizer configuration.'}


def selection_properties(seqs):
    import seqme as sm
    p=properties(seqs).astype(float)
    p[:,1]=sm.models.Charge()(seqs)
    p[:,3]=sm.models.HydrophobicMoment()(seqs)
    return p


def chemistry_diagnostics(seqs):
    # Transparent warning indicators only; no invented synthesis pass/fail threshold.
    longest=[]; fractions=[]; c=[]; oxidation=[]
    hydrophobic=set('AILMFWVY')
    for s in seqs:
        runs=[];last=None;run=0
        for a in s:
            run=run+1 if a==last else 1;last=a;runs.append(run)
        longest.append(max(runs));fractions.append(sum(a in hydrophobic for a in s)/len(s))
        c.append(s.count('C'));oxidation.append(sum(s.count(a) for a in 'CMW'))
    return {'longest_identical_run_q90':float(np.quantile(longest,.9)),
        'hydrophobic_fraction_q90':float(np.quantile(fractions,.9)),
        'cysteine_containing_fraction':float((np.asarray(c)>0).mean()),
        'odd_cysteine_count_fraction':float((np.asarray(c)%2==1).mean()),
        'CMW_count_q90':float(np.quantile(oxidation,.9)),
        'validated_synthesizability_pass_rate':None,
        'warning':'Descriptive sequence indicators only; these do not validate synthesis or justify hard exclusions.'}


def load_extra_embeddings(path,seqs):
    with np.load(path,allow_pickle=False) as d:
        strings=d['sequences'].astype(str).tolist(); e=np.asarray(d['embeddings'],dtype=np.float32)
        name=str(d['model'].item())
    if len(strings)!=len(set(strings)) or e.ndim!=2 or len(e)!=len(strings) or not np.isfinite(e).all():raise ValueError('Invalid external embedding file')
    index={s:i for i,s in enumerate(strings)}
    absent=set(seqs)-index.keys()
    if absent:raise ValueError(f'External embeddings missing {len(absent)} required sequences')
    return e[[index[s] for s in seqs]],name


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/pilot');p.add_argument('--device',default='cuda',choices=['cuda','cpu','auto'])
    p.add_argument('--models',nargs='+',default=['flow','ar'],choices=['flow','ar'])
    p.add_argument('--variants',nargs='+',default=['euler_16','temp_1.0','temp_1.1'])
    p.add_argument('--select-n',type=int,default=1250);p.add_argument('--eval-n',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--swaps',type=int,default=250)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--esm-model',default='esm2_t6_8M_UR50D',choices=['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D'])
    p.add_argument('--esm-batch',type=int,default=32)
    p.add_argument('--external-scores');p.add_argument('--score-schema')
    p.add_argument('--generic-reference',help='Real generic peptide FASTA for evaluation only; not shuffled AMPs')
    p.add_argument('--extra-embeddings',help='Aligned NPZ of real ESM-C embeddings from isolated exporter')
    p.add_argument('--mmseqs',help='Optional path/name of installed MMseqs2 executable')
    p.add_argument('--report-dir',help='Separate experiment output directory')
    p.add_argument('--property-weight',type=float,default=.5)
    p.add_argument('--marginal-weight',type=float,default=0.)
    p.add_argument('--redundancy-weight',type=float,default=0.)
    p.add_argument('--evaluation-seed',type=int,default=141)
    p.add_argument('--export-only',action='store_true',help='Prepare predictor and embedding FASTAs without evaluating')
    a=p.parse_args(argv)
    if min(a.select_n,a.eval_n)<4 or min(a.repeats,a.workers,a.esm_batch)<1 or a.swaps<0:p.error('Positive counts required')
    if bool(a.external_scores)!=bool(a.score_schema):p.error('Supply both --external-scores and --score-schema')
    with threadpool_limits(limits=a.workers):run(a)


def run(a):
    seed_all(a.seed)
    out=Path(a.out);root=Path(a.report_dir) if a.report_dir else out/'phase1';root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text())
    for name,h in manifest['outputs'].items():
        if digest(out/name)!=h:raise ValueError(f'Prepared file changed: {name}')
    dev=read_fasta(out/'dev.fasta',clean=False);ref=read_fasta(out/'audit.fasta',clean=False)
    validate(dev,'development reference');validate(ref,'audit reference')
    if set(dev)&set(ref):raise ValueError('Development/audit sequence overlap')
    known=sorted(set(read_fasta(out/'all_positive.fasta'))|set(read_fasta(out/'forbidden.fasta',clean=False)));set_known=set(known)
    pools={}
    for model in a.models:
        for variant in a.variants:
            dest=out/model/variant
            if (dest/'eligible.fasta').exists():
                seqs=read_fasta(dest/'eligible.fasta',clean=False);validate(seqs,str(dest));pools[f'{model}/{variant}']=seqs
    if not pools:raise ValueError('No requested existing generator pools')
    pools['pooled']=sorted(set(s for seqs in pools.values() for s in seqs))
    union=pools['pooled'];write_fasta(root/'scoring_input.fasta',union)
    generic=read_fasta(a.generic_reference,clean=False) if a.generic_reference else []
    if generic:validate(generic,'generic reference')
    required=sorted(set(union+dev+ref+generic));write_fasta(root/'embedding_input.fasta',required)
    config=vars(a).copy();config.update(candidate_pool_hashes={k:digest(out/k/'eligible.fasta') for k in pools if k!='pooled'},
        manifest_sha256=digest(out/'manifest.json'))
    json_write(root/'configuration.json',config)
    if a.export_only:print(f'Exported {len(union)} scoring sequences and {len(required)} embedding sequences to {root}');return
    started=time.perf_counter();proxy,proxy_report=train_proxy(out,a)
    local=proxy.predict_proba(features(union))[:,1] if proxy else np.zeros(len(union))
    table=None;models=[]
    if a.external_scores:
        utility,score_report,table,models=external_scores(a.external_scores,a.score_schema,union)
        score_report.update(csv_sha256=digest(a.external_scores),schema_sha256=digest(a.score_schema))
    else:
        # Small weight and separate fidelity preset: local score cannot stand in for independent activity.
        utility=ranks(local);score_report={'source':'local proxy only','warning':'Independent AMP/MIC evaluation missing; screening is provisional.'}
    scores=dict(zip(union,utility));local_scores=dict(zip(union,local))
    print('Measure known-sequence similarity and sampled pool redundancy',flush=True)
    maxima=[]
    selection_known=sorted(set(known)-set(ref))
    for start in range(0,len(union),64):
        maxima.extend((process.cdist(union[start:start+64],selection_known,scorer=fuzz.ratio,workers=a.workers,dtype=np.float32).max(1)/100).tolist())
    novelty=dict(zip(union,ranks(1-np.asarray(maxima))))
    # Soft pool-density penalty, measured against a fixed uniform candidate sample.
    density_sample=[union[i] for i in np.random.default_rng(a.seed).choice(len(union),min(1024,len(union)),replace=False)]
    crowd=[]
    for start in range(0,len(union),128):
        chunk=union[start:start+128]
        similarities=process.cdist(chunk,density_sample,scorer=fuzz.ratio,workers=a.workers,dtype=np.float32)
        crowd.extend([float(((row>80).sum()-int(s in density_sample))/len(density_sample)) for s,row in zip(chunk,similarities)])
    crowding=dict(zip(union,ranks(np.asarray(crowd))))
    cache=ESMCache(out,a)
    report={'status':'DEVELOPMENT_PHASE1_SURROGATES_NOT_OFFICIAL_SCORE','configuration':config,
        'activity':score_report,'local_proxy_audit':proxy_report,
        'missing':['Hidden organizer weights/reference sets','Validated synthesizability constraints','Organizer alignment normalization/search settings'],
        'warning':'Audit split has already been inspected; now development evidence, not an untouched final test.',
        'complete':False,'sets':{},'reference_diagnostics':{}}
    if not a.mmseqs:report['missing'].append('Alignment-based novelty measurement')
    if not table:report['missing'].append('Independent AMP/MIC predictions')
    if not generic:report['missing'].append('Generic peptide reference')
    if not a.extra_embeddings:report['missing'].append('ESM-C embeddings')
    json_write(root/'report.json',report)
    try:
        all_e=cache.embed(required);lookup={s:i for i,s in enumerate(required)}
        emb=lambda seqs:all_e[[lookup[s] for s in seqs]]
        de=emb(dev);re=emb(ref)
        report['reference_diagnostics']['esm2']=reference_floor(re,a.eval_n,a.repeats,a.seed)
        extra=None
        if a.extra_embeddings:
            extra,extra_name=load_extra_embeddings(a.extra_embeddings,required)
            if not extra_name.startswith('esmc_'):raise ValueError('Expected real ESM-C embedding model metadata')
        for name,seqs in pools.items():
            if len(seqs)<a.select_n:raise ValueError(f'{name}: only {len(seqs)} candidates for {a.select_n} selection')
            print(f'Select/evaluate {name}: {len(seqs)} candidates',flush=True)
            e=emb(seqs);u=np.array([scores[s] for s in seqs]);lp=np.array([local_scores[s] for s in seqs])
            selector=CoverageSelector(e,de,selection_properties(seqs),selection_properties(dev),a.seed,a.property_weight,a.marginal_weight,a.redundancy_weight)
            random_ids=np.random.default_rng(a.seed).choice(len(seqs),a.select_n,replace=False)
            old_ids=select(seqs,lp,dev,a.select_n,a.seed)
            policies={'raw':(np.arange(len(seqs)),{}),'random':(random_ids,{}),'legacy':(old_ids,{})}
            for policy,weight in [('fidelity',0.),('balanced',.005),('activity',.02)]:
                policies[policy]=selector.choose(seqs,u,a.select_n,weight,a.swaps)
            novelty_u=np.array([.75*scores[s]+.25*novelty[s]-.1*crowding[s] for s in seqs])
            policies['novelty']=selector.choose(seqs,novelty_u,a.select_n,.005,a.swaps)
            policies['novelty'][1]['utility_definition']='0.75 activity rank utility + 0.25 known-novelty percentile - 0.1 sampled-pool crowding percentile'
            entry={}
            for policy,(ids,selection) in policies.items():
                chosen=[seqs[i] for i in ids];path=root/name/policy/'library.fasta';write_fasta(path,chosen)
                m={'count':len(chosen),'fasta_sha256':digest(path),'selection':selection,
                    'activity_rank_utility_mean':float(u[ids].mean()),
                    'esm2_amp':embedding_metrics(e[ids],re,de,a.eval_n,a.repeats,a.evaluation_seed),
                    'seqme_properties':seqme_properties(chosen,ref),
                    'sequence':sequence_metrics(chosen,known,a.eval_n,a.evaluation_seed,a.workers),
                    'chemistry_diagnostics':chemistry_diagnostics(chosen)}
                # Full-set coverage of development embedding/property regions.
                occupied=np.unique(selector.bins[ids]);m['development_region_coverage']=float(selector.weights[occupied].sum())
                if table:
                    m['external_models']={}
                    for model in models:
                        values=np.array([float(table[s][model['column']]) for s in chosen])
                        stats={'mean':float(values.mean()),'median':float(np.median(values)),'coverage':1.,'family':model['family'],'kind':model['kind']}
                        if model['kind']=='mic_um':stats['predicted_fraction_le_16uM']=float((values<=16).mean())
                        m['external_models'][model['column']]=stats
                if generic:m['esm2_generic']=embedding_metrics(e[ids],emb(generic),emb(generic),a.eval_n,a.repeats,a.evaluation_seed)
                if extra is not None:
                    ix=lambda strings:extra[[lookup[s] for s in strings]]
                    m['extra_embedding_model']=extra_name
                    m['esmc_amp']=embedding_metrics(ix(chosen),ix(ref),ix(dev),a.eval_n,a.repeats,a.evaluation_seed)
                    if generic:m['esmc_generic']=embedding_metrics(ix(chosen),ix(generic),ix(generic),a.eval_n,a.repeats,a.evaluation_seed)
                if a.mmseqs:
                    from .alignment import alignment_novelty
                    sample=[chosen[i] for i in np.random.default_rng(a.seed).choice(len(chosen),min(a.eval_n,len(chosen)),replace=False)]
                    m['alignment_novelty']=alignment_novelty(sample,known,path.parent/'alignment',a.mmseqs,a.workers)
                entry[policy]=m
            report['sets'][name]=entry;json_write(root/'report.json',report)
        report['seconds']=time.perf_counter()-started;report['complete']=True;json_write(root/'report.json',report)
        lines=['# Phase 1 selector comparison','','No official score or automatic winning policy is claimed.','',
            '| Pool / policy | Count | AMP FBD ↓ | AMP MMD ↓ | Precision ↑ | Recall ↑ | Known ratio ≤0.8 ↑ | Conformity ↑ | Rank utility ↑ |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for pool,policies in report['sets'].items():
            for policy,m in policies.items():
                e=m['esm2_amp']['mean'];s=m['sequence']
                lines.append(f"| {pool}/{policy} | {m['count']} | {e['fbd']:.4f} | {e['mmd_unbiased']:.4f} | {e['precision']:.3f} | {e['recall']:.3f} | {s['known_indel_ratio_le_0_8_fraction']:.3f} | {m['seqme_properties']['conformity_mean']:.3f} | {m['activity_rank_utility_mean']:.3f} |")
        lines+=['','Raw has a different total set size; all embedding comparisons use equal-size subsamples.','',
            'Missing: '+', '.join(report['missing'])+'.','',
            'Selection fits development references only. Audit properties and embeddings are used for reporting. Repeated audit use makes this development evidence.','',
            'The generic-reference metrics are diagnostics, not an instruction to maximize distance from generic peptides. Hidden metric directions and weights are unknown.','',
            'Compare all policies; do not choose solely by FBD or rank utility. Read subsample variation, seqme conformity, per-model activity, and sequence redundancy in report.json.']
        (root/'report.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        report['error']=str(exc);json_write(root/'report.json',report);raise
    finally:cache.close()
    print(f'Finished: {root}/report.md',flush=True)

if __name__=='__main__':main()
