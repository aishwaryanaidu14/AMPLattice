from pathlib import Path
import json
import time
import urllib.request
from collections import Counter
import numpy as np
from rapidfuzz import process, fuzz
from .common import AA, read_fasta, write_fasta, json_write, digest

URLS = {
 'amps.fasta':'https://raw.githubusercontent.com/szczurek-lab/OmegAMP/main/data/generative-model-data/AMPs.fasta',
 'negatives.fasta':'https://raw.githubusercontent.com/szczurek-lab/OmegAMP/main/data/activity-data/curated-Non-AMPs.fasta',
 'forbidden.fasta':'https://raw.githubusercontent.com/szczurek-lab/amp-challenge-2027/main/data/antibacterial.fasta',
}

def fetch(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'downloads.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else {}
    for name,url in URLS.items():
        p=folder/name
        if p.exists() and name in old:
            if digest(p)!=old[name]['sha256']: raise ValueError(f'Download checksum changed: {p}')
            continue
        print(f'Download {name}',flush=True)
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'amp-decision-pilot/0.1'})
                with urllib.request.urlopen(req,timeout=120) as r: content=r.read()
                if not content.lstrip().startswith(b'>'): raise ValueError(f'Not FASTA: {url}')
                tmp=Path(str(p)+'.tmp'); tmp.write_bytes(content); tmp.replace(p)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2*(attempt+1))
        old[name]={'url':url,'sha256':digest(p),'bytes':p.stat().st_size}
        json_write(manifest,old)
    return folder

def components(seqs, threshold=80.0, workers=4):
    """Exact connected components of ALL pairs with normalized Indel similarity >= threshold.
    Bounded block memory; O(n^2) CPU work. Not an alignment-based identity metric.
    """
    n=len(seqs); parent=np.arange(n)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=int(parent[i])
        return i
    for start in range(0,n,256):
        d=process.cdist(seqs[start:start+256],seqs,scorer=fuzz.ratio,
                        score_cutoff=threshold,workers=workers,dtype=np.float32)
        for row in range(len(d)):
            i=start+row
            for j in np.flatnonzero(d[row,i+1:]>=threshold)+i+1:
                ri,rj=root(i),root(int(j))
                if ri!=rj: parent[rj]=ri
        if start%2048==0: print(f'Family split: {min(start+256,n)}/{n}',flush=True)
    return np.array([root(i) for i in range(n)])

def prepare(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if not args.fasta:
        source=fetch(Path(args.data_dir)); pos_path=source/'amps.fasta'
        neg_path=source/'negatives.fasta'; forbidden_path=source/'forbidden.fasta'
    else:
        pos_path=Path(args.fasta)
        neg_path=Path(args.negatives) if args.negatives else None
        if args.reference: forbidden_path=Path(args.reference)
        else:
            source=fetch(Path(args.data_dir)); forbidden_path=source/'forbidden.fasta'
    signature={'positive_sha256':digest(pos_path),'negative_sha256':digest(neg_path) if neg_path else None,
               'forbidden_sha256':digest(forbidden_path),'split_seed':args.seed,'cluster_ratio':args.cluster_ratio}
    mp=out/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        if old['signature']!=signature: raise ValueError('Data/config differs from existing run. Choose a new --out.')
        for name,h in old['outputs'].items():
            if not (out/name).exists() or digest(out/name)!=h: raise ValueError(f'Missing/modified split file {name}; use new --out')
        print('Reuse verified data split',flush=True); return out
    pos=read_fasta(pos_path); neg=read_fasta(neg_path) if neg_path else []
    # A peptide may have conflicting assay-specific labels. Drop contradictions from proxy, keep positive sequence training.
    conflicts=set(pos)&set(neg); neg=sorted(set(neg)-set(pos))
    seqs=pos+neg
    if len(pos)<100: raise ValueError(f'Only {len(pos)} eligible AMPs; need >=100 for this pilot.')
    groups=components(seqs,args.cluster_ratio,args.workers)
    rng=np.random.default_rng(args.seed); g=np.unique(groups); rng.shuffle(g)
    # Assign whole connected components, largest first with deterministic tie order.
    g=sorted(g,key=lambda x:-int(np.sum(groups==x)))
    counts=np.zeros((3,2)); target=np.array([.8,.1,.1])[:,None]*[len(pos),max(1,len(neg))]
    split=np.zeros(len(seqs),dtype=int)
    for group in g:
        ids=np.flatnonzero(groups==group); add=np.array([(ids<len(pos)).sum(),(ids>=len(pos)).sum()])
        # Minimize normalized total squared target deviation after this group's assignment.
        scores=[]
        for k in range(3):
            c=counts.copy(); c[k]+=add; scores.append(np.sum((c-target)**2/np.maximum(target,1)))
        k=int(np.argmin(scores)); split[ids]=k; counts[k]+=add
    for k,name in enumerate(['train','dev','audit']):
        a=[s for i,s in enumerate(seqs) if split[i]==k and i<len(pos)]
        b=[s for i,s in enumerate(seqs) if split[i]==k and i>=len(pos)]
        if len(a)<10: raise ValueError(f'Too few {name} AMPs ({len(a)}), giant family may dominate. Supply broader data.')
        write_fasta(out/f'{name}.fasta',a); write_fasta(out/f'{name}_negative.fasta',b)
    write_fasta(out/'all_positive.fasta',pos)
    reference=sorted({s.upper() for s in read_fasta(forbidden_path,clean=False) if s and set(s.upper())<=set(AA)})
    write_fasta(out/'forbidden.fasta',reference)
    files=[p for p in out.glob('*.fasta')]
    json_write(mp,{'signature':signature,'positive_source':str(pos_path),'negative_source':str(neg_path),
        'counts':{'positive':len(pos),'negative':len(neg),'conflicting_negative_removed':len(conflicts),
                  'split_train_dev_audit_pos_neg':counts.astype(int).tolist(),'families':len(g)},
        'chemistry':'Sequence-only corpus; free-terminus/linear assay provenance not established. No potency labels inferred.',
        'outputs':{p.name:digest(p) for p in files}})
    return out
