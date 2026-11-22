"""TCN + noncausal Transformer with separate mean and disagreement heads."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class MUSEDU(nn.Module):
    def __init__(self, in_features:int, hidden:int=96, layers:int=2, heads:int=4, dropout:float=.1,
                 descriptor_features:int=18):
        super().__init__()
        if in_features<descriptor_features or hidden%heads:
            raise ValueError('Invalid dimensions')
        self.projection=nn.Sequential(nn.Linear(in_features,hidden),nn.LayerNorm(hidden))
        self.local=nn.Sequential(
            nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU(),
            nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU())
        layer=nn.TransformerEncoderLayer(hidden,heads,hidden*2,dropout,batch_first=True,
                                          activation='gelu',norm_first=False)
        self.context=nn.TransformerEncoder(layer,num_layers=layers,enable_nested_tensor=False)
        self.fuse=nn.Sequential(nn.Linear(hidden*2,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.mean_head=nn.Linear(hidden,2)
        self.dis_head=nn.Linear(hidden,2)
        self.hidden_dim=hidden

    def encode(self,x,time,valid):
        """Shared v1/v2 audio and temporal backbone."""
        # Bounded Fourier time encodings; actual absolute time is preserved in the dataset.
        z=self.projection(x)
        phase=time.unsqueeze(-1)/torch.exp(torch.linspace(0,8,self.hidden_dim//2,device=x.device))
        positional=torch.cat([phase.sin(),phase.cos()],-1)
        z=z+positional[:,:,:z.shape[-1]]
        z=z.masked_fill(~valid.unsqueeze(-1),0)
        local=self.local(z.transpose(1,2)).transpose(1,2)
        context=self.context(z,src_key_padding_mask=~valid)
        h=self.fuse(torch.cat((local,context),dim=-1))
        return h

    def forward(self,x,time,valid):
        """Inputs B,T,F; B,T seconds; valid mask B,T for padding."""
        h=self.encode(x,time,valid)
        return {'mean':torch.tanh(self.mean_head(h)),
                'tau':F.softplus(self.dis_head(h))+1e-4,'hidden':h}


def center_valid(x,valid):
    """Subtract each recording's valid-frame mean without using target labels."""
    weight=valid.unsqueeze(-1).to(x.dtype)
    center=(x*weight).sum(1,keepdim=True)/weight.sum(1,keepdim=True).clamp_min(1)
    return (x-center)*weight


class MUSEDUV2(MUSEDU):
    """Shared backbone with song-level and centered local emotion/SD branches."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        del self.mean_head,self.dis_head
        self.global_mean_head=nn.Linear(self.hidden_dim,2)
        self.local_mean_head=nn.Linear(self.hidden_dim,2)
        self.global_dis_head=nn.Linear(self.hidden_dim,2)
        self.local_dis_head=nn.Linear(self.hidden_dim,2)

    def forward(self,x,time,valid):
        h=self.encode(x,time,valid)
        weight=valid.unsqueeze(-1).to(h.dtype)
        pooled=(h*weight).sum(1)/weight.sum(1).clamp_min(1)
        global_mean=self.global_mean_head(pooled)
        local_mean=center_valid(self.local_mean_head(h),valid)
        global_dis=self.global_dis_head(pooled)
        local_dis=center_valid(self.local_dis_head(h),valid)
        mean=torch.tanh(global_mean[:,None,:]+local_mean)*weight
        tau=(F.softplus(global_dis[:,None,:]+local_dis)+1e-4)*weight
        return {'mean':mean,'tau':tau,'hidden':h,
                'song_mean':torch.tanh(global_mean),
                'song_tau':F.softplus(global_dis)+1e-4,
                'local_residual':local_mean,'local_dispersion':local_dis}


def _masked_song_means(y,mask):
    weight=mask.to(y.dtype)
    return (y*weight).sum(1)/weight.sum(1).clamp_min(1)


def _equal_song_huber(pred,target,mask,delta=.1):
    if not mask.any():return pred.sum()*0
    elem=F.huber_loss(pred,target,delta=delta,reduction='none')*mask
    count=mask.sum((1,2))
    return (elem.sum((1,2))/count.clamp_min(1))[count>0].mean()


def centered_correlation_loss(pred,target,mask,eps=1e-6):
    """Equal-song/axis Pearson loss with separately stabilized vector norms.

    Adding epsilon to the *product of energies* distorts correlation for small
    trajectories. Normalize each centered vector instead; eps is a norm floor.
    The previous target-energy exclusion rule is retained for comparability.
    """
    weight=mask.to(pred.dtype)
    count=weight.sum(1)
    yp=(pred-_masked_song_means(pred,mask)[:,None,:])*weight
    yt=(target-_masked_song_means(target,mask)[:,None,:])*weight
    target_energy=(yt*yt).sum(1)
    valid=(count>=3)&(target_energy>eps)
    if not valid.any():return pred.sum()*0
    corr=(F.normalize(yp,p=2,dim=1,eps=eps)*
          F.normalize(yt,p=2,dim=1,eps=eps)).sum(1)
    return (1-corr[valid]).mean()


def legacy_centered_correlation_loss(pred,target,mask,eps=1e-6):
    """Original energy-product epsilon, retained to reproduce older experiments."""
    weight=mask.to(pred.dtype);count=weight.sum(1)
    yp=(pred-_masked_song_means(pred,mask)[:,None,:])*weight
    yt=(target-_masked_song_means(target,mask)[:,None,:])*weight
    target_energy=(yt*yt).sum(1);pred_energy=(yp*yp).sum(1)
    valid=(count>=3)&(target_energy>eps)
    if not valid.any():return pred.sum()*0
    corr=(yp*yt).sum(1)/torch.sqrt(pred_energy*target_energy+eps)
    return (1-corr[valid]).mean()


def multiscale_change_loss(pred,target,mask,times,step_seconds,lags=(4,10,20)):
    """Masked 2/5/10-second changes, requiring every intermediate bin."""
    expected=torch.as_tensor(step_seconds,device=times.device,dtype=times.dtype)
    if expected.numel()==1:expected=expected.reshape(1,1)
    else:expected=expected.reshape(-1,1)
    adjacent=torch.isclose(times[:,1:]-times[:,:-1],expected,rtol=.01,atol=1e-4)
    losses=[]
    for lag in lags:
        n=times.shape[1]-lag
        if n<=0:continue
        valid=mask[:,lag:]&mask[:,:-lag]
        for offset in range(lag):
            valid=valid&adjacent[:,offset:offset+n].unsqueeze(-1)
        for offset in range(1,lag):
            valid=valid&mask[:,offset:offset+n]
        if valid.any():
            losses.append(_equal_song_huber(pred[:,lag:]-pred[:,:-lag],
                            target[:,lag:]-target[:,:-lag],valid))
    return torch.stack(losses).mean() if losses else pred.sum()*0


V2_LOSS_WEIGHTS={'point':1.,'song':.3,'shape':.02,'multi':.3,'dis':.2}


def v2_supervised_loss(out,batch,weights=None,eps=1e-3):
    weights=V2_LOSS_WEIGHTS if weights is None else weights
    if set(weights)!=set(V2_LOSS_WEIGHTS):raise ValueError('v2 loss needs point/song/shape/multi/dis weights')
    y=batch['mean'];mask=batch['mask'].bool()
    point=_equal_song_huber(out['mean'],y,mask)
    song_targets=_masked_song_means(y,mask)
    song_valid=mask.any(1)
    song=masked_huber(out['song_mean'],song_targets,song_valid)
    shape=centered_correlation_loss(out['mean'],y,mask)
    multi=multiscale_change_loss(out['mean'],y,mask,batch['times'],batch['step'])
    std_mask=batch['std_mask'].bool()&mask
    dis=_equal_song_huber(torch.log(out['tau']+eps),
                          torch.log(batch['std'].clamp_min(eps)+eps),std_mask)
    components={'point':point,'song':song,'shape':shape,'multi':multi,'dis':dis}
    total=sum(float(weights[k])*v for k,v in components.items())
    return total,{k:float(v.detach()) for k,v in components.items()}


def masked_huber(pred,target,mask,delta=.1):
    if not mask.any():return pred.sum()*0
    return F.huber_loss(pred[mask],target[mask],delta=delta,reduction='mean')


def supervised_loss(out, batch, transition_weight=.25, dispersion_weight=.2,eps=1e-3):
    y=batch['mean']; mask=batch['mask']
    mean=masked_huber(out['mean'],y,mask)
    dt=batch['times'][:,1:]-batch['times'][:,:-1]
    expected=batch['step']
    if torch.is_tensor(expected) and expected.ndim==1: expected=expected[:,None]
    adjacent=(dt>0)&torch.isclose(dt,expected,rtol=.01,atol=1e-4)
    dm=mask[:,1:]&mask[:,:-1]&adjacent.unsqueeze(-1)
    trans=masked_huber(out['mean'][:,1:]-out['mean'][:,:-1],
                        y[:,1:]-y[:,:-1],dm)
    std_mask=batch['std_mask']&mask
    # reference zero SD is permissible but logarithmic loss requires a common positive floor.
    disp=masked_huber(torch.log(out['tau']+eps),torch.log(batch['std'].clamp_min(eps)+eps),std_mask)
    total=mean+transition_weight*trans+dispersion_weight*disp
    return total,{'mean':float(mean.detach()),'transition':float(trans.detach()),
                  'dispersion':float(disp.detach())}


class ErrorScale(nn.Module):
    """Positive Laplace residual scale conditioned on transitions, tau, ensemble variance."""
    def __init__(self, hidden:int, width:int=48):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden+2+2+2,width),nn.GELU(),nn.Linear(width,2))

    def forward(self, hidden,tau,ensemble_std,mean,time,step_seconds=None,
                use_transition=True,use_tau=True,use_ensemble_std=True):
        diff=torch.zeros_like(mean)
        d_h=torch.zeros_like(hidden[:,:,:1])
        raw_step=time[:,1:]-time[:,:-1]
        if step_seconds is None:
            # Fallback for callers without metadata; production pipelines pass it.
            candidate=torch.where(raw_step>0,raw_step,torch.full_like(raw_step,float('inf')))
            expected=candidate.amin(dim=1,keepdim=True)
        else:
            specified=torch.as_tensor(step_seconds,device=time.device,dtype=time.dtype)
            if specified.numel()==1: expected=specified.reshape(1,1)
            elif specified.numel()==len(time): expected=specified.reshape(-1,1)
            else: raise ValueError('step_seconds must be scalar or one value per song')
        adjacent=(raw_step>0)&torch.isclose(raw_step,expected,rtol=.01,atol=1e-4)
        step=raw_step.clamp_min(1e-3)
        diff[:,1:]=torch.where(adjacent.unsqueeze(-1),(mean[:,1:]-mean[:,:-1])/step.unsqueeze(-1),0.)
        d_h[:,1:]=torch.where(adjacent.unsqueeze(-1),
            ((hidden[:,1:]-hidden[:,:-1])/step.unsqueeze(-1)).norm(dim=-1,keepdim=True),0.)
        transition=torch.cat((diff.norm(dim=-1,keepdim=True),d_h),-1)
        if not use_transition: transition=torch.zeros_like(transition)
        if not use_tau: tau=torch.zeros_like(tau)
        if not use_ensemble_std: ensemble_std=torch.zeros_like(ensemble_std)
        inp=torch.cat((hidden.detach(),tau.detach(),ensemble_std.detach(),transition.detach()),-1)
        return F.softplus(self.net(inp))+1e-4


class HiddenErrorScale(nn.Module):
    """Optional v2 error scale using the frozen point model's hidden states only."""
    def __init__(self,hidden:int,width:int=48):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden,width),nn.GELU(),nn.Linear(width,2))
    def forward(self,hidden):
        return F.softplus(self.net(hidden.detach()))+1e-4


def laplace_scale_loss(scale,pred,target,mask):
    residual=(pred.detach()-target).abs()
    elem=torch.log(scale)+residual/scale
    # All valid bins within a song receive equal total song weight.
    per_song=(elem*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    present=mask.any((1,2))
    if not present.any():return scale.sum()*0
    return per_song[present].mean()
