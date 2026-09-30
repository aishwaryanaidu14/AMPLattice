import os
# Must be set before initializing CUDA, including console entrypoint imports.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
import torch
from .common import digest
from .data import prepare
from .train import train
from .generate import generate
from .evaluate import evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description='AMP flow/AR decision pilot; default run downloads small FASTAs, trains, samples and audits.')
    p.add_argument('command',choices=['run','prepare','train','generate','evaluate'],nargs='?',default='run')
    p.add_argument('--out',default='runs/pilot')
    p.add_argument('--data-dir',default='data/raw')
    p.add_argument('--fasta',help='Own positive AMP FASTA; accepts only canonical sequences of length 8-50')
    p.add_argument('--negatives',help='Optional labelled negatives with --fasta. Do not use shuffled sequences as true negatives.')
    p.add_argument('--reference',help='Optional pinned competition reference with --fasta')
    p.add_argument('--models',nargs='+',choices=['flow','ar'],default=['flow','ar'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cluster-ratio',type=float,default=80)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--steps',type=int,default=3000,help='Max optimizer steps per model (can increase to resume)')
    p.add_argument('--minutes',type=float,default=30,help='Cumulative training time budget PER model, not total run time')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--checkpoint-every',type=int,default=200)
    p.add_argument('--mixed-precision',action='store_true',help='Optional CUDA float16, disabled by default for stability')
    p.add_argument('--samples',type=int,default=5000,help='Raw draws PER sampling variant')
    p.add_argument('--sample-batch',type=int,default=128)
    p.add_argument('--flow-steps',nargs='+',type=int,default=[16,32])
    p.add_argument('--ar-temperatures',nargs='+',type=float,default=[1.0,1.1])
    p.add_argument('--esm',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--esm-model',choices=['esm2_t6_8M_UR50D','esm2_t12_35M_UR50D'],default='esm2_t6_8M_UR50D')
    p.add_argument('--esm-batch',type=int,default=32)
    p.add_argument('--eval-n',type=int,default=1000,help='Equal-sized sample cap per embedding metric')
    args=p.parse_args(argv)
    if args.width%args.heads: p.error('--width must be divisible by --heads')
    for key in ['steps','batch_size','sample_batch','esm_batch','eval_n','workers','checkpoint_every','layers','width','heads']:
        if getattr(args,key)<1: p.error(f'{key} must be positive')
    if args.samples<16 or args.minutes<=0 or args.lr<=0: p.error('Need samples>=16, minutes>0, lr>0')
    if not 0<args.cluster_ratio<=100: p.error('--cluster-ratio must be in (0,100]')
    if any(x<1 for x in args.flow_steps) or any(x<=0 for x in args.ar_temperatures): p.error('Sampling steps and temperatures must be positive')
    if args.negatives and not args.fasta: p.error('--negatives requires --fasta')
    if args.reference and not args.fasta: p.error('--reference requires --fasta')
    torch.set_num_threads(args.workers)
    print(f'PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}; output={args.out}',flush=True)
    if args.command in ('run','prepare'): prepare(args)
    if args.command=='prepare': return
    if not (Path(args.out)/'manifest.json').exists(): p.error('Run prepare or run first.')
    manifest=json.loads((Path(args.out)/'manifest.json').read_text())
    for name,h in manifest['outputs'].items():
        path=Path(args.out)/name
        if not path.exists() or digest(path)!=h: p.error(f'Modified/missing prepared data: {path}')
    if args.command in ('run','train'):
        for kind in args.models: train(args,kind)
    if args.command=='train': return
    dirs=[]
    for kind in args.models:
        variants=[f'euler_{k}' for k in args.flow_steps] if kind=='flow' else [f'temp_{t}' for t in args.ar_temperatures]
        for variant in variants:
            dest=Path(args.out)/kind/variant
            if args.command in ('run','generate'): generate(args,kind,variant)
            if not (dest/'generation.json').exists(): p.error(f'Missing {dest}; run generation with matching settings.')
            dirs.append(dest)
    if args.command in ('run','evaluate'): evaluate(args,dirs)

if __name__=='__main__':
    main()
