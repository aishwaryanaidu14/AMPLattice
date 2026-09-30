from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
INDEX = {a:i for i,a in enumerate(AA)}
PAD, BOS = 20, 21

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sid(s):
    return hashlib.sha256(s.encode()).hexdigest()

def read_fasta(path, clean=True):
    seqs, pieces = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('>'):
            if pieces: seqs.append(''.join(pieces)); pieces = []
        else:
            pieces.append(line)
    if pieces: seqs.append(''.join(pieces))
    if clean:
        seqs = [s.upper() for s in seqs]
        seqs = [s for s in seqs if 8 <= len(s) <= 50 and set(s) <= set(AA)]
        return sorted(set(seqs))
    return seqs

def write_fasta(path, seqs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+'.tmp')
    with tmp.open('w') as f:
        for i,s in enumerate(seqs): f.write(f'>p{i:08d}\n{s}\n')
    tmp.replace(path)

def json_write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def device_for(name):
    import torch
    if name == 'auto': name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Check torch installation and NVIDIA driver.')
    return torch.device(name)

def tokens(seqs):
    arr = np.full((len(seqs),50), PAD, dtype=np.int64)
    for i,s in enumerate(seqs): arr[i,:len(s)] = [INDEX[a] for a in s]
    return arr

# Five published residue scales, attributed in NOTICE.md.
SCALES = json.loads(Path(__file__).with_name('scales.json').read_text())
RAW_CODE = np.array([[SCALES[k][a] for k in SCALES] for a in AA], dtype=np.float32)
CODE = (RAW_CODE-RAW_CODE.mean(0))/RAW_CODE.std(0)
EIS = dict(zip('ARNDCQEGHILKMFPSTWYV', [0.62,-2.53,-0.78,-0.90,0.29,-0.85,-0.74,0.48,-0.40,1.38,1.06,-1.50,0.64,1.19,0.12,-0.18,-0.05,0.81,0.26,1.08]))

def properties(seqs):
    """L, approximate HH net charge at pH7 free termini, Eisenberg mean, full-chain 100deg moment.
    Explicit diagnostic definitions, NOT asserted identical to organizer seqme defaults.
    """
    out = []
    for s in seqs:
        q = 1/(1+10**(7-8.0)) - 1/(1+10**(3.1-7))
        for aa,pk in [('K',10.5),('R',12.5),('H',6.0)]: q += s.count(aa)/(1+10**(7-pk))
        for aa,pk in [('D',3.9),('E',4.1),('C',8.3),('Y',10.1)]: q -= s.count(aa)/(1+10**(pk-7))
        h = np.array([EIS[a] for a in s]); angle = np.arange(len(s))*np.deg2rad(100)
        moment = abs(np.sum(h*np.exp(1j*angle)))/len(s)
        out.append([len(s), q, float(h.mean()), float(moment)])
    return np.asarray(out,dtype=np.float32).reshape(-1,4)

def features(seqs):
    """Transparent local activity-proxy features, no external pretrained classifier."""
    f = np.zeros((len(seqs),424),np.float32); f[:,:4] = properties(seqs)
    for i,s in enumerate(seqs):
        for a in s: f[i,4+INDEX[a]] += 1/len(s)
        for a,b in zip(s,s[1:]): f[i,24+20*INDEX[a]+INDEX[b]] += 1/max(1,len(s)-1)
    return f
