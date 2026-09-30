"""Run verified BATTLE-AMP scoring interface in its separately installed environment.
Run this with the Python environment that contains snakemake, with conda available.
Does not install models or reinterpret missing predictions as inactivity.
"""
import argparse,json,subprocess,csv,os,sys,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--fasta',required=True);p.add_argument('--output',required=True)
    p.add_argument('--models',default='amplify,mbc-attention');p.add_argument('--cores',type=int,default=4)
    a=p.parse_args();repo=Path(a.repo).resolve();output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    requested=a.models.split(',')
    if 'ampeppy' in requested and len(requested)>1:
        # Separate processes ensure CPU-only AMPeppy cannot hide CUDA from other models.
        parts=[]
        for label,names,cpu in [('cpu',['ampeppy'],True),('gpu',[m for m in requested if m!='ampeppy'],False)]:
            target=output.with_name(output.stem+'.'+label+'.csv')
            env=os.environ.copy()
            if cpu:env['CUDA_VISIBLE_DEVICES']=''
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--repo',str(repo),'--fasta',str(Path(a.fasta).resolve()),'--output',str(target),'--models',','.join(names),'--cores',str(a.cores)],env=env,check=True)
            parts.append(target)
        merged={};columns=['sequence'];schemas=[];reports={};expected_sequences=None
        for target in parts:
            with target.open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames or []
                if 'sequence' not in fields:raise RuntimeError('Missing sequence column')
                overlap=(set(fields)-{'sequence'}) & set(columns)
                if overlap:raise RuntimeError('Duplicate predictor columns across device runs')
                columns.extend(c for c in fields if c!='sequence');seen=set()
                for row in reader:
                    seq=row['sequence'].strip().upper()
                    if seq in seen:raise RuntimeError('Duplicate score sequence')
                    seen.add(seq);merged.setdefault(seq,{'sequence':seq}).update({k:v for k,v in row.items() if k!='sequence'})
                if expected_sequences is None:expected_sequences=seen
                elif seen!=expected_sequences:raise RuntimeError('CPU/GPU prediction sequence sets differ')
            schemas.extend(json.loads(target.with_suffix('.schema.json').read_text())['models'])
            reports[target.name]=json.loads(target.with_suffix('.report.json').read_text())
        with output.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            for seq in sorted(merged):writer.writerow(merged[seq])
        output.with_suffix('.schema.json').write_text(json.dumps({'models':schemas},indent=2)+'\n')
        output.with_suffix('.report.json').write_text(json.dumps({'device_policy':'AMPeppy: CUDA hidden; remaining models: inherited GPU visibility','component_reports':reports},indent=2)+'\n')
        print('Merged CPU/GPU predictions:',output);return
    if requested==['ampeppy']:os.environ['CUDA_VISIBLE_DEVICES']=''
    snake=shutil.which('snakemake')
    if not snake:raise RuntimeError('Snakemake not found in the BATTLE environment')
    command=[snake,'--profile','profile/','--cores',str(a.cores),'score','--config',
        'fasta='+str(Path(a.fasta).resolve()),'run_models='+a.models,'output='+str(output),'unit=uM']
    battle_env=os.environ.copy()
    # BATTLE launches its own Snakemake-managed Conda model environments.
    # Do not let uv's virtualenv marker make BATTLE believe one is stacked.
    battle_env.pop('VIRTUAL_ENV',None)
    battle_env.pop('PYTHONHOME',None)
    virtual=os.environ.get('VIRTUAL_ENV')
    if virtual:
        bad=str(Path(virtual)/'bin')
        battle_env['PATH']=os.pathsep.join(x for x in battle_env.get('PATH','').split(os.pathsep) if str(Path(x))!=bad)
    battle_env['PATH']=str(Path(snake).parent)+os.pathsep+battle_env.get('PATH','')
    result=subprocess.run(command,cwd=repo,env=battle_env,check=True)
    report_path=output.with_suffix('.report.json')
    if not output.exists() or not report_path.exists():raise RuntimeError(f'BATTLE output/report missing; exit status {result.returncode}')
    report=json.loads(report_path.read_text())
    if report.get('status')!='ok':raise RuntimeError('BATTLE report is not ok: '+str(report_path))
    statuses=report.get('models')
    if isinstance(statuses,dict):
        failed=[m for m in requested if statuses.get(m,{}).get('status')!='ok']
        if failed:raise RuntimeError('BATTLE requested model failed: '+', '.join(failed))
    print('Read predictor statuses in',report_path)
    with output.open(newline='') as f:columns=csv.DictReader(f).fieldnames or []
    models=[]
    for c in columns:
        if c.endswith('_prob'):kind='amp_probability';variant=c[:-5]
        elif c.endswith('_MIC_uM'):kind='mic_um';variant=c[:-7]
        else:continue
        family=variant
        for prefix in ['APEX','Deep-AMP','HydrAMP','SenseXAMP']:
            if variant.startswith(prefix):family=prefix;break
        models.append({'column':c,'family':family,'kind':kind,'target':variant})
    if not models:raise RuntimeError('No recognized BATTLE prediction columns; inspect output/report')
    expected={'ampeppy':'ampeppy_prob','amplify':'amplify_prob','mbc-attention':'mbc-attention_MIC_uM'}
    missing=[expected[m] for m in a.models.split(',') if m in expected and expected[m] not in columns]
    if missing:raise RuntimeError('Requested BATTLE models missing output columns: '+', '.join(missing))
    wanted=[expected[m] for m in requested if m in expected]
    # Some BATTLE output tables carry a stale/additional predictor column.
    # Keep only the requested models so CPU and GPU tables merge cleanly.
    if wanted:
        rows=[]
        with output.open(newline='') as f:
            for row in csv.DictReader(f):rows.append({'sequence':row['sequence'],**{c:row[c] for c in wanted}})
        with output.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['sequence',*wanted]);writer.writeheader();writer.writerows(rows)
        models=[m for m in models if m['column'] in wanted]
    schema=output.with_suffix('.schema.json');schema.write_text(json.dumps({'models':models},indent=2)+'\n')
    print('Score schema:',schema)
    print('The evaluator rejects missing requested scores. Inspect the BATTLE report even if Snakemake succeeded.')
if __name__=='__main__':main()
