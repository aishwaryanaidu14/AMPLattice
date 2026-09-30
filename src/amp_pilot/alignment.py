"""Sampled MMseqs2 local alignment novelty surrogate; not organizer configuration."""
import subprocess
from pathlib import Path
import numpy as np
from .common import write_fasta,digest


def parse_bits(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        q,t,b=line.split('\t');b=float(b)
        if not np.isfinite(b) or b<0:raise ValueError('Invalid alignment bit score')
        rows.append((q,t,b))
    return rows


def alignment_novelty(seqs,known,folder,executable='mmseqs',workers=4):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    query=folder/'query.fasta';reference=folder/'known.fasta'
    write_fasta(query,seqs);write_fasta(reference,known)
    settings=['--search-type','1','--alignment-mode','3','-s','7.5','-e','1000',
        '-k','6','--spaced-kmer-mode','0','--mask','0','--comp-bias-corr','0','--min-ungapped-score','0',
        '--max-seqs','10000','--threads',str(workers),'--format-output','query,target,bits']
    for target,name in [(query,'self'),(reference,'known')]:
        subprocess.run([executable,'easy-search',str(query),str(target),str(folder/(name+'.tsv')),
            str(folder/(name+'_tmp')),*settings,*(['--add-self-matches','1'] if name=='self' else [])],check=True,stdout=subprocess.DEVNULL)
    self_bits={q:b for q,t,b in parse_bits(folder/'self.tsv') if q==t}
    ids=[f'p{i:08d}' for i in range(len(seqs))]
    if set(ids)-self_bits.keys():raise RuntimeError('Missing MMseqs2 self alignment scores; cannot normalize')
    maxima={q:0. for q in ids}
    for q,t,b in parse_bits(folder/'known.tsv'):
        maxima[q]=max(maxima[q],b/self_bits[q])
    ratios=np.array([maxima[q] for q in ids])
    return {'sample_size':len(seqs),'query_normalized_max_bits_median':float(np.median(ratios)),
        'novelty_one_minus_clipped_max_bits_mean':float((1-ratios.clip(0,1)).mean()),
        'query_sha256':digest(query),'known_sha256':digest(reference),
        'settings':settings,'self_search_add_self_matches':True,'warning':'Heuristic MMseqs2 search; normalized by query self bits. No returned hit means zero similarity under this search. Not the organizer novelty definition or exact exhaustive alignment.'}
