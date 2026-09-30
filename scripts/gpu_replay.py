"""Run two independent empty-work GPU replays and write concise evidence."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--evidence',type=Path,default=Path('verification-work'))
    p.add_argument('--runs',type=int,default=2)
    a=p.parse_args();root=a.root.resolve();evidence=a.evidence.resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    expected=json.loads((root/'assets/expected.json').read_text())
    report={'fresh_replays':[],'passed':False}
    for i in range(a.runs):
        work=evidence/f'run-{i+1}';out=evidence/f'output-{i+1}'
        if work.exists() or out.exists():raise RuntimeError('Use a new evidence directory; reusing work is not fresh verification')
        command=['uv','run','--frozen','generate','--work-dir',str(work),'--output-dir',str(out)]
        started=time.monotonic()
        with (evidence/f'run-{i+1}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        item={'command':command,'exit':result.returncode,'seconds':round(time.monotonic()-started,2)}
        for key,path in {'pool':work/'production/candidates.fasta','scores':work/'production/scores.csv','baseline':work/'production/baseline.fasta','hybrid':work/'hybrid/hybrid/library.fasta','library':out/'library.fasta','top':out/'top.fasta'}.items():
            if path.exists():item[key]={'sha256':sha(path),'matches_original':sha(path)==expected[key]}
        report['fresh_replays'].append(item)
        (evidence/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
        if result.returncode:sys.exit(result.returncode)
    report['passed']=a.runs>=2 and all(all(item.get(k,{}).get('matches_original',False) for k in ['pool','scores','baseline','hybrid','library','top']) for item in report['fresh_replays'])
    # Byte comparison additionally checks the two complete independent outputs.
    first=evidence/'output-1'
    report['byte_identical']=all((first/n).read_bytes()==(evidence/f'output-{i+1}'/n).read_bytes() for i in range(1,a.runs) for n in ['library.fasta','top.fasta'])
    report['passed']=report['passed'] and report['byte_identical']
    (evidence/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:sys.exit(1)

if __name__=='__main__':main()
