"""Resumable generator + selector experiments; public surrogate recommendations only."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .selector import ranks


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def summarize(reports):
    """Rank settings across seeds, equally weighting metric families, then minimax regret.

    Quantile ranks are relative to this batch. Weights are declared sensitivity
    scenarios, never inferred organizer weights. Raw/random/legacy are controls.
    """
    groups={}
    for setting,path,report in reports:
        for pool,policies in report['sets'].items():
            for policy,m in policies.items():
                if policy in ('raw','random','legacy'):continue
                groups.setdefault((setting,pool,policy),[]).append((path,m))
    if not groups:raise ValueError('No complete optimized experiments')
    keys=sorted(groups);families={};families['distribution']=[];families['properties']=[];families['sequence']=[]
    activity={}
    for key in keys:
        vals=[m for _,m in groups[key]]
        mean=lambda f:float(np.mean([f(m) for m in vals]))
        families['distribution'].append([mean(lambda m:-m['esm2_amp']['mean']['fbd']),mean(lambda m:-m['esm2_amp']['mean']['mmd_unbiased']),mean(lambda m:m['esm2_amp']['mean']['precision']),mean(lambda m:m['esm2_amp']['mean']['recall'])])
        families['properties'].append([mean(lambda m:-m['seqme_properties']['wasserstein_std_units']['charge']),mean(lambda m:-m['seqme_properties']['wasserstein_std_units']['hydrophobic_moment'])])
        families['sequence'].append([mean(lambda m:m['sequence']['known_indel_ratio_le_0_8_fraction']),mean(lambda m:-m['sequence']['sample_internal_near_duplicate_fraction']),mean(lambda m:m['sequence']['pairwise_indel_diversity'])])
        if not all(m.get('external_models') for m in vals):raise ValueError('Independent predictions required for recommendation')
        for col,info in vals[0]['external_models'].items():
            family=info['family'];kind=info['kind']
            if kind=='amp_probability':
                activity.setdefault(family,{}).setdefault(col,[]).append(mean(lambda m,c=col:m['external_models'][c]['mean']))
            else:
                for metric in ['median','mean']:
                    activity.setdefault(family,{}).setdefault(col+':'+metric,[]).append(mean(lambda m,c=col,k=metric:-np.log2(m['external_models'][c][k])))
                activity.setdefault(family,{}).setdefault(col+':fraction_le_16',[]).append(mean(lambda m,c=col:m['external_models'][c]['predicted_fraction_le_16uM']))
    for name,arr in list(families.items()):
        a=np.asarray(arr);families[name]=np.mean(np.column_stack([ranks(a[:,j]) for j in range(a.shape[1])]),axis=1)
    votes=[]
    for family,columns in activity.items():
        votes.append(np.mean(np.stack([ranks(x) for x in columns.values()]),axis=0))
    families['activity']=np.mean(votes,axis=0)
    order=['activity','distribution','properties','sequence'];scores=np.column_stack([families[n] for n in order])
    scenarios=np.array([x for x in itertools.product([.1,.2,.3,.4],repeat=4) if abs(sum(x)-1)<1e-9]+[(.25,)*4]+[tuple(.55 if j==i else .15 for j in range(4)) for i in range(4)])
    scenario_scores=scores@scenarios.T;regret=scenario_scores.max(0)-scenario_scores
    rows=[]
    for i,key in enumerate(keys):
        dominated=any(np.all(scores[j]>=scores[i]-1e-12) and np.any(scores[j]>scores[i]+1e-12) for j in range(len(keys)) if j!=i)
        paths=[str(p.parent/key[1]/key[2]/'library.fasta') for p,_ in groups[key]]
        fbd=[m['esm2_amp']['mean']['fbd'] for _,m in groups[key]]
        rows.append({'setting':key[0],'pool':key[1],'policy':key[2],'seeds':len(groups[key]),'pareto':not dominated,'max_scenario_regret':float(regret[i].max()),'equal_family_score':float(scores[i].mean()),'family_scores':dict(zip(order,map(float,scores[i]))),'fbd_mean':float(np.mean(fbd)),'fbd_seed_sd':float(np.std(fbd)),'libraries':paths})
    rows.sort(key=lambda x:(x['max_scenario_regret'],-x['equal_family_score'],x['setting'],x['pool'],x['policy']))
    return {'winner':rows[0],'ranking':rows,'weight_scenarios':scenarios.tolist(),'metric_families':order,'warning':'Development surrogate recommendation; hidden weights, references, synthesis constraints and repeated audit use prevent an official winner claim. The same predictor families guide selection and evaluation, so predictor exploitation remains possible. Quantile ranks depend on the compared batch. Seed SD uses repeated selection runs, not independent training runs.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/pilot');p.add_argument('--device',default='cuda',choices=['cuda','cpu','auto'])
    p.add_argument('--selector-only',action='store_true',help='Use existing candidates and existing BATTLE scores')
    p.add_argument('--battle-repo',default=str(Path.home()/'battleamp-snakemake'))
    p.add_argument('--snakemake',help='Absolute snakemake executable from the working BATTLE environment')
    p.add_argument('--samples',type=int,default=5000)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    p.add_argument('--swaps',type=int,default=2000);p.add_argument('--eval-n',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--external-scores');p.add_argument('--score-schema')
    a=p.parse_args(argv)
    if a.samples<16 or min(a.swaps,a.eval_n,a.repeats,a.workers)<1:p.error('Positive experiment counts required')
    if bool(a.external_scores)!=bool(a.score_schema):p.error('Supply scores and schema together')
    import torch
    torch.set_num_threads(a.workers)
    from .common import digest,json_write,read_fasta,write_fasta
    from .phase1 import main as evaluate
    out=Path(a.out).resolve();root=out/'experiments_v3';root.mkdir(parents=True,exist_ok=True)
    snakemake=None
    if not a.selector_only:
        repo=Path(a.battle_repo).expanduser().resolve()
        options=[a.snakemake,shutil.which('snakemake'),str(repo/'.venv/bin/snakemake'),str(repo/'venv/bin/snakemake')]
        for directory in [Path.home()/'.venvs',Path.home()/'miniconda3/envs',Path.home()/'anaconda3/envs']:
            options.extend(str(x) for x in sorted(directory.glob('*/bin/snakemake')))
        snakemake=next((str(Path(x).expanduser().resolve()) for x in options if x and Path(x).expanduser().is_file()),None)
        if not snakemake or not (repo/'profile').exists():
            p.error('Generator experiments need the working BATTLE environment. Pass --snakemake /path/to/its/bin/snakemake and --battle-repo, or use --selector-only for cached selector experiments.')
    variants=['euler_16','temp_1.0','temp_1.1']
    if not a.selector_only:
        from types import SimpleNamespace
        from .generate import generate
        experiments=[('flow','euler_32'),('flow','euler_64'),('flow','heun_32'),('flow','blend35_heun_32'),('ar','temp_0.9'),('ar','temp_1.2')]
        for kind,variant in experiments:
            print('GENERATOR EXPERIMENT',kind,variant,flush=True)
            generate(SimpleNamespace(out=str(out),samples=a.samples,sample_batch=128,seed=42,device=a.device),kind,variant)
            variants.append(variant)
    base=['--out',str(out),'--device',a.device,'--variants',*variants,'--workers',str(a.workers),'--select-n','1250','--eval-n',str(a.eval_n),'--repeats',str(a.repeats),'--swaps',str(a.swaps),'--evaluation-seed','141']
    export=root/'inputs';evaluate(base+['--report-dir',str(export),'--export-only'])
    scoring=export/'scoring_input.fasta'
    if a.selector_only:
        score=Path(a.external_scores) if a.external_scores else out/'phase1/battle_scores.csv'
        schema=Path(a.score_schema) if a.score_schema else out/'phase1/battle_scores.schema.json'
    else:
        score=root/'battle_scores.csv';schema=root/'battle_scores.schema.json';stamp=root/'battle_complete.json'
        signature={'fasta':digest(scoring),'models':'ampeppy,amplify,mbc-attention','repo':str(repo),'snakemake':snakemake}
        valid=stamp.exists() and score.exists() and schema.exists()
        if valid:
            old=json.loads(stamp.read_text());valid=old.get('signature')==signature and old.get('score_hash')==digest(score) and old.get('schema_hash')==digest(schema)
        if not valid:
            env=os.environ.copy();env['PATH']=str(Path(snakemake).parent)+os.pathsep+env.get('PATH','')
            helper=Path(__file__).resolve().parents[1]/'run_battle.py'
            subprocess.run([sys.executable,str(helper),'--repo',str(repo),'--fasta',str(scoring),'--output',str(score),'--cores',str(a.workers)],env=env,check=True)
            from .phase1 import external_scores
            external_scores(score,schema,read_fasta(scoring,clean=False))
            json_write(stamp,{'signature':signature,'score_hash':digest(score),'schema_hash':digest(schema)})
    if not score.exists() or not schema.exists():raise FileNotFoundError('BATTLE score CSV/schema missing')
    # Strict coverage check BEFORE expensive selector/evaluation runs.
    from .phase1 import external_scores
    external_scores(score,schema,read_fasta(scoring,clean=False))
    configs=[('baseline',.5,0.,0.),('property',1.,.1,.005),('strong_property',2.,.3,.01)]
    reports=[]
    for name,pw,mw,rw in configs:
        for seed in dict.fromkeys(a.seeds):
            dest=root/name/f'seed_{seed}';reportpath=dest/'report.json'
            command=base+['--external-scores',str(score),'--score-schema',str(schema),'--report-dir',str(dest),'--seed',str(seed),'--property-weight',str(pw),'--marginal-weight',str(mw),'--redundancy-weight',str(rw)]
            sig=fingerprint({'args':command,'input':digest(scoring),'score':digest(score),'schema':digest(schema),'manifest':digest(out/'manifest.json'),'code':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}})
            stamp=dest/'complete.json';reuse=False
            if stamp.exists() and reportpath.exists():
                old=json.loads(stamp.read_text());reuse=old.get('signature')==sig and old.get('report_hash')==digest(reportpath)
                if reuse:
                    saved=json.loads(reportpath.read_text());reuse=saved.get('complete',False) and all((dest/pool/policy/'library.fasta').exists() and digest(dest/pool/policy/'library.fasta')==m['fasta_sha256'] for pool,policies in saved['sets'].items() for policy,m in policies.items())
            if not reuse:
                print('SELECTOR EXPERIMENT',name,'seed',seed,flush=True);evaluate(command)
                json_write(stamp,{'signature':sig,'report_hash':digest(reportpath)})
            reports.append((name,reportpath,json.loads(reportpath.read_text())))
            recommendation=summarize(reports);json_write(root/'recommendation.partial.json',recommendation)
    recommendation=summarize(reports)
    recommendation['complete']=True;recommendation['configuration']=vars(a)
    recommendation['missing']=reports[0][2]['missing']
    recommendation['generator_experiments']='Existing checkpoints; integration, descriptor shrinkage, and AR temperature. No new training or new model architecture.'
    json_write(root/'recommendation.json',recommendation)
    winner=recommendation['winner'];shutil.copyfile(winner['libraries'][0],root/'recommended.fasta')
    with (root/'ranking.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['setting','pool','policy','seeds','pareto','max_scenario_regret','equal_family_score','fbd_mean','fbd_seed_sd']);writer.writeheader()
        for row in recommendation['ranking']:writer.writerow({k:row[k] for k in writer.fieldnames})
    lines=['# Automatic experiment recommendation','',f"Recommended development setting: **{winner['pool']} / {winner['policy']} / {winner['setting']}**.",'',recommendation['warning'],'','The exported FASTA uses the first requested selection seed; settings are ranked using all requested seeds. It is a 1,250-sequence pilot, not a complete 50,000-sequence submission.','','| Rank | Generator pool | Selector | Property setting | Worst scenario regret | FBD | Seed SD |','|---|---|---|---|---:|---:|---:|']
    for i,row in enumerate(recommendation['ranking'][:15],1):lines.append(f"| {i} | {row['pool']} | {row['policy']} | {row['setting']} | {row['max_scenario_regret']:.4f} | {row['fbd_mean']:.4f} | {row['fbd_seed_sd']:.4f} |")
    lines+=['','Missing evaluations: '+', '.join(recommendation['missing']), '', 'Generator variants have similar draw budgets, but pooled selection has a larger candidate budget; its gains do not establish a better standalone generator. Conformity and chemistry indicators remain reported diagnostics rather than invented synthesis pass/fail rules.']
    (root/'recommendation.md').write_text('\n'.join(lines)+'\n')
    print('\nCOMPLETE:',root/'recommendation.md','\nRECOMMENDED FASTA:',root/'recommended.fasta',flush=True)

if __name__=='__main__':main()
