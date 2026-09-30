import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .common import read_fasta, properties, tokens, seed_all, device_for, json_write, digest
from .models import Generator, loss_for


def save_checkpoint(path,payload):
    path=Path(path); tmp=Path(str(path)+'.tmp'); torch.save(payload,tmp); tmp.replace(path)

def tensor_data(seqs,mean,scale,device):
    prop=properties(seqs)
    return (torch.tensor(tokens(seqs),device=device),torch.tensor(prop[:,0],device=device,dtype=torch.long),
            torch.tensor((prop-mean)/scale,device=device))

def train(args,kind):
    out=Path(args.out); model_dir=out/kind; model_dir.mkdir(exist_ok=True)
    device=device_for(args.device); seed_all(args.seed); torch.set_num_threads(args.workers)
    train_seq=read_fasta(out/'train.fasta'); dev_seq=read_fasta(out/'dev.fasta')
    p=properties(train_seq); mean=p.mean(0); scale=p.std(0).clip(.01)
    spec={'kind':kind,'width':args.width,'layers':args.layers,'heads':args.heads}
    signature={'spec':spec,'data':digest(out/'manifest.json'),'seed':args.seed,'batch':args.batch_size,
               'lr':args.lr,'mixed_precision':args.mixed_precision,'device_type':device.type}
    model=Generator(**spec).to(device); ema=copy.deepcopy(model).eval()
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    amp=args.mixed_precision and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    latest=model_dir/'latest.pt'; best=model_dir/'best.pt'; start=0; elapsed=0; best_loss=float('inf')
    rng=np.random.default_rng(args.seed)
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False)
        if ck['signature']!=signature: raise ValueError(f'Incompatible training resume {kind}. Choose new --out.')
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema']); optim.load_state_dict(ck['optim'])
        scaler.load_state_dict(ck['scaler']); rng.bit_generator.state=ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        start=ck['step']; elapsed=ck['seconds']; best_loss=ck['best_loss']
        print(f'Resume {kind}: step={start}, cumulative={elapsed:.1f}s',flush=True)
    td=tensor_data(train_seq,mean,scale,device)
    # Dev model selection only. Audit split is never used for checkpoint choice.
    dv=dev_seq[:min(1024,len(dev_seq))]; vd=tensor_data(dv,mean,scale,device)
    began=time.perf_counter(); last_print=began; losses=[]
    def checkpoint(step):
        nonlocal best_loss
        ema.eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(args.seed+777)
            vals=[]
            with torch.inference_mode():
                for a in range(0,len(dv),args.batch_size):
                    chunk=[x[a:a+args.batch_size] for x in vd]
                    vals.append((loss_for(ema,*chunk,training=False).item(),len(chunk[0])))
            val=sum(v*n for v,n in vals)/sum(n for _,n in vals)
        seconds=elapsed+time.perf_counter()-began
        if not np.isfinite(val): raise FloatingPointError('Nonfinite validation loss; rerun without --mixed-precision or lower --lr in a new output directory.')
        if val<best_loss:
            best_loss=val
            save_checkpoint(best,{'spec':spec,'model':ema.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),
                                  'step':step,'val_loss':val,'signature':signature})
        save_checkpoint(latest,{'signature':signature,'model':model.state_dict(),'ema':ema.state_dict(),
                'optim':optim.state_dict(),'scaler':scaler.state_dict(),'step':step,'seconds':seconds,
                'best_loss':best_loss,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
        print(f'{kind}: step {step}, train={np.mean(losses):.4f}, dev={val:.4f}, elapsed={seconds/60:.1f}min',flush=True)
        with (model_dir/'training.jsonl').open('a') as f:
            f.write(json.dumps({'step':step,'train_loss':float(np.mean(losses)),'dev_loss':val,'seconds':seconds})+'\n')
    step=start
    try:
        for step in range(start+1,args.steps+1):
            if elapsed+time.perf_counter()-began>=args.minutes*60 and (losses or best.exists()):
                step-=1; break
            model.train(); ids=rng.integers(len(train_seq),size=args.batch_size)
            batch=[x[ids] for x in td]; optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                loss=loss_for(model,*batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optim); scaler.update()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()): ep.lerp_(mp,.01)
            losses.append(loss.item())
            if len(losses)>100: losses.pop(0)
            if step%args.checkpoint_every==0: checkpoint(step)
            elif time.perf_counter()-last_print>30:
                print(f'{kind}: step {step}/{args.steps}, recent loss={np.mean(losses):.4f}',flush=True); last_print=time.perf_counter()
        if step>start: checkpoint(step)
    except KeyboardInterrupt:
        print('Interrupted; preserving current training state.',flush=True)
        if losses: checkpoint(step)
        raise
    if not best.exists(): raise RuntimeError('No checkpoint produced. Increase --steps/--minutes.')
    ck=torch.load(latest,map_location='cpu',weights_only=False)
    json_write(model_dir/'training_summary.json',{'kind':kind,'steps':ck['step'],'seconds':ck['seconds'],
        'parameters':sum(p.numel() for p in model.parameters()),'best_dev_loss':ck['best_loss'],
        'device':str(device),'cuda_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'torch':torch.__version__,'mixed_precision':amp,
        'warning':'Flow MSE and AR cross entropy have different meanings; never compare their magnitudes.'})
    return best
