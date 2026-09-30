import math
import torch
from torch import nn
from .common import PAD, BOS, CODE

class Block(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.n1=nn.LayerNorm(width); self.n2=nn.LayerNorm(width)
        self.attn=nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,pad,causal=False):
        h=self.n1(x)
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1) if causal else None
        # need_weights=True uses reproducible math attention on supported torch versions.
        a,_=self.attn(h,h,h,key_padding_mask=pad,attn_mask=mask,need_weights=True)
        x=x+a; return x+self.ff(self.n2(x))

class Generator(nn.Module):
    def __init__(self,kind='flow',width=256,layers=4,heads=4):
        super().__init__(); self.kind=kind
        self.register_buffer('code',torch.tensor(CODE))
        self.register_buffer('freq',torch.exp(torch.linspace(0,math.log(1000),16)))
        self.input=nn.Linear(5,width) if kind=='flow' else nn.Embedding(22,width)
        self.position=nn.Parameter(torch.randn(1,50,width)*.02)
        self.condition=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width))
        self.time=nn.Sequential(nn.Linear(32,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([Block(width,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(width); self.output=nn.Linear(width,5 if kind=='flow' else 20)
    def forward(self,x,lengths,c,t=None,drop_condition=False):
        # Condition is normalized (length, charge, hydrophobicity, moment).
        present=torch.ones_like(c)
        if drop_condition:
            present[:,1:]=(torch.rand_like(c[:,1:])>.10).float()
        h=self.input(x); n=h.shape[1]
        h=h+self.position[:,:n]+self.condition(torch.cat([c*present,present],-1))[:,None]
        if self.kind=='flow':
            phase=t[:,None]*self.freq[None]
            h=h+self.time(torch.cat([phase.sin(),phase.cos()],-1))[:,None]
        pad=torch.arange(n,device=h.device)[None]>=lengths[:,None]
        for b in self.blocks: h=b(h,pad,causal=self.kind=='ar')
        return self.output(self.norm(h))

def loss_for(model,tok,lengths,c,training=True):
    mask=torch.arange(tok.shape[1],device=tok.device)[None]<lengths[:,None]
    if model.kind=='flow':
        x1=model.code[tok.clamp_max(19)]; x0=torch.randn_like(x1)
        t=torch.rand(len(tok),device=tok.device)
        xt=((1-t[:,None,None])*x0+t[:,None,None]*x1)*mask[:,:,None]
        pred=model(xt,lengths,c,t,drop_condition=training)
        return (((pred-(x1-x0))**2)*mask[:,:,None]).sum()/(mask.sum()*5)
    inp=torch.full_like(tok,BOS); inp[:,1:]=tok[:,:-1]
    logits=model(inp,lengths,c,drop_condition=training)
    return nn.functional.cross_entropy(logits.flatten(0,1),tok.flatten(),ignore_index=PAD)

@torch.inference_mode()
def sample(model,lengths,c,steps=32,temperature=1.0,solver="euler"):
    b=len(lengths); n=int(lengths.max()); device=lengths.device
    mask=torch.arange(n,device=device)[None]<lengths[:,None]
    if model.kind=='flow':
        x=torch.randn(b,n,5,device=device)*mask[:,:,None]
        dt=1/steps
        for k in range(steps):
            t=torch.full((b,),k/steps,device=device)
            v=model(x,lengths,c,t)
            if solver=="heun":
                proposed=(x+dt*v)*mask[:,:,None]
                endpoint=torch.full((b,),(k+1)/steps,device=device)
                v=(v+model(proposed,lengths,c,endpoint))/2
            elif solver!="euler":raise ValueError("Unknown flow solver")
            x=(x+dt*v)*mask[:,:,None]
        distance=((x[:,:,None]-model.code[None,None])**2).sum(-1)
        tok=distance.argmin(-1)
        uncertainty=distance.min(-1).values[mask].mean().item()
    else:
        inp=torch.full((b,1),BOS,device=device,dtype=torch.long); result=[]
        for k in range(n):
            # Prefix recomputation baseline, no KV cache: timings explicitly label this implementation.
            logits=model(inp,lengths,c)[:,-1]/temperature
            nxt=torch.multinomial(logits.softmax(-1),1); result.append(nxt)
            inp=torch.cat([inp,nxt],1)
        tok=torch.cat(result,1); uncertainty=None
    return tok,uncertainty
