from pathlib import Path
import json
import time
import numpy as np
import torch
from .common import AA, properties, read_fasta, write_fasta, seed_all, device_for, digest, json_write
from .models import Generator, sample


def generate(args,kind,variant):
    out=Path(args.out); dest=out/kind/variant; dest.mkdir(parents=True,exist_ok=True)
    checkpoint=out/kind/'best.pt'; meta=dest/'generation.json'
    blend=float(variant.split('_')[0][5:])/100 if variant.startswith('blend') else 0.
    solver='heun' if 'heun_' in variant else 'euler'
    if not 0<=blend<=1:raise ValueError('Invalid condition blend')
    settings={'generator_revision':2,'condition_blend':blend,'solver':solver,'checkpoint_sha256':digest(checkpoint),'samples':args.samples,'batch':args.sample_batch,
              'seed':args.seed+1000,'variant':variant,'device':str(device_for(args.device))}
    condition_file=getattr(args,'condition_fasta',None)
    if condition_file:settings['condition_fasta_sha256']=digest(condition_file)
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['settings']==settings and all((dest/x).exists() and digest(dest/x)==h for x,h in old['files'].items()):
            print(f'Reuse {kind}/{variant}',flush=True); return dest
    device=device_for(args.device); seed_all(args.seed+1000)
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    model=Generator(**ck['spec']).to(device); model.load_state_dict(ck['model']); model.eval()
    train=read_fasta(condition_file,clean=False) if condition_file else read_fasta(out/'train.fasta'); props=properties(train)
    rng=np.random.default_rng(args.seed+1000)
    # Empirical targets from training data by default; explicit development templates for production replenishment.
    targets=props[rng.integers(len(props),size=args.samples)]
    mean=np.array(ck['mean']); scale=np.array(ck['scale'])
    # Shrink descriptor conditions toward training mean, retaining empirical lengths.
    targets[:,1:]=(1-blend)*targets[:,1:]+blend*mean[1:]
    raw=[]; distances=[]; started=time.perf_counter()
    for start in range(0,args.samples,args.sample_batch):
        target=targets[start:start+args.sample_batch]
        lengths=torch.tensor(target[:,0],device=device,dtype=torch.long)
        c=torch.tensor((target-mean)/scale,device=device,dtype=torch.float32)
        if kind=='flow': tok,err=sample(model,lengths,c,steps=int(variant.split('_')[-1]),solver=solver)
        else: tok,err=sample(model,lengths,c,temperature=float(variant.split('_')[-1]))
        arr=tok.cpu().numpy()
        raw += [''.join(AA[j] for j in arr[i,:int(n)]) for i,n in enumerate(target[:,0])]
        if err is not None: distances.append(err)
        if start//args.sample_batch%10==0: print(f'{kind}/{variant}: generated {len(raw)}/{args.samples}',flush=True)
    if device.type=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    all_known=set(read_fasta(out/'all_positive.fasta'))
    forbidden=set(read_fasta(out/'forbidden.fasta',clean=False))
    # Stronger exact-removal policy for this pilot: all supplied AMP data plus official reference.
    valid=list(dict.fromkeys(s for s in raw if s not in all_known and s not in forbidden))
    write_fasta(dest/'raw.fasta',raw); write_fasta(dest/'eligible.fasta',valid)
    np.save(dest/'target_properties.npy',targets)
    actual=properties(raw)
    json_write(meta,{'settings':settings,'draws':len(raw),'unique':len(set(raw)),
        'unique_eligible':len(valid),'seconds':seconds,'raw_per_second':len(raw)/seconds,
        'eligible_per_second':len(valid)/seconds,'exact_known_fraction':float(np.mean([s in all_known or s in forbidden for s in raw])),
        'normalized_property_mae':np.mean(abs(actual-targets)/scale,axis=0).tolist(),
        'codebook_squared_distance':float(np.mean(distances)) if distances else None,
        'implementation':solver+' full-sequence updates' if kind=='flow' else 'causal AR with prefix recomputation (no KV cache)',
        'files':{p.name:digest(p) for p in [dest/'raw.fasta',dest/'eligible.fasta',dest/'target_properties.npy']}})
    return dest
