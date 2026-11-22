"""Four neural reference models for dynamic valence/arousal (paper-inspired)."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self,dim,hidden=64,dropout=.2):
        super().__init__()
        self.pre=nn.LayerNorm(dim)
        self.encoder=nn.LSTM(dim,hidden,num_layers=2,dropout=dropout,batch_first=True)
        self.out=nn.Linear(hidden,2)
    def forward(self,x,valid):
        lengths=valid.sum(1).cpu().clamp_min(1)
        seq=nn.utils.rnn.pack_padded_sequence(self.pre(x),lengths,batch_first=True,enforce_sorted=False)
        z,_=self.encoder(seq)
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        return self.out(z).tanh().masked_fill(~valid[:,:,None],0)


class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.3),
         nn.Linear(128,64),nn.ReLU(),nn.Dropout(.3),
         nn.Linear(64,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,2))
    def forward(self,x,valid):return torch.tanh(self.net(x)).masked_fill(~valid[:,:,None],0)


class MultiLayerProbe(nn.Module):
    def __init__(self,dim,n_layers,dropout=.25):
        super().__init__()
        self.weights=nn.Parameter(torch.zeros(n_layers))
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,2))
    def forward(self,x,valid):
        if x.ndim!=4:raise ValueError('Expected [B,T,L,D] layer embeddings')
        z=(x*self.weights.softmax(0)[None,None,:,None]).sum(2)
        return torch.tanh(self.net(z)).masked_fill(~valid[:,:,None],0)


class ConformerBlock(nn.Module):
    """Macaron FFNs, relative-free attention, depthwise GLU conv.

    Practical 2-Hz log-mel adaptation, NOT original model's precise setup.
    """
    def __init__(self,dim=128,heads=4,dropout=.15):
        super().__init__()
        def ff():return nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.SiLU(),nn.Dropout(dropout),nn.Linear(dim*4,dim))
        self.ff1=ff();self.ff2=ff()
        self.normatt=nn.LayerNorm(dim)
        self.att=nn.MultiheadAttention(dim,heads,batch_first=True,dropout=dropout)
        self.normconv=nn.LayerNorm(dim)
        self.conv1=nn.Conv1d(dim,dim*2,1)
        self.dw=nn.Conv1d(dim,dim,15,padding=7,groups=dim)
        self.bn=nn.BatchNorm1d(dim)
        self.conv2=nn.Conv1d(dim,dim,1)
        self.final=nn.LayerNorm(dim)
    def forward(self,x,valid):
        x=x+.5*self.ff1(x)
        q=self.normatt(x)
        att,_=self.att(q,q,q,key_padding_mask=~valid,need_weights=False)
        x=x+att
        v=self.normconv(x).masked_fill(~valid[:,:,None],0).transpose(1,2)
        v=F.glu(self.conv1(v),dim=1)
        v=self.conv2(F.silu(self.bn(self.dw(v)))).transpose(1,2)
        x=x+v
        return self.final(x+.5*self.ff2(x)).masked_fill(~valid[:,:,None],0)


class Conformer(nn.Module):
    def __init__(self,dim,hidden=128,blocks=2):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden))
        self.blocks=nn.ModuleList([ConformerBlock(hidden) for _ in range(blocks)])
        self.head=nn.Linear(hidden,2)
    def forward(self,x,valid):
        h=self.proj(x)
        # Continuous sample-position encoding for 0.5-second annotation grid.
        pos=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[None,:,None]
        freqs=torch.exp(torch.linspace(0,7,h.shape[-1]//2,device=x.device,dtype=x.dtype))
        phase=pos/freqs
        h=h+torch.cat((phase.sin(),phase.cos()),dim=-1)
        for layer in self.blocks:h=layer(h,valid)
        return torch.tanh(self.head(h)).masked_fill(~valid[:,:,None],0)


def create_model(kind,shape):
    dim=shape[-1]
    if kind=='lstm':return LSTM(dim)
    if kind=='mlp':return MLP(dim)
    if kind=='probe':
        if len(shape)!=2:raise ValueError('Probe expects (layers, hidden)')
        return MultiLayerProbe(dim,shape[0])
    if kind=='conformer':return Conformer(dim)
    raise ValueError(kind)
