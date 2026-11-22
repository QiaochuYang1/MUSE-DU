"""Exploratory v2 uncertainty comparison with uniform bands as default."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .experiment import load_manifest,read_npz,nominal_step
from .v2_experiment import load_v2_checkpoint,predict_v2_ensemble,_reference_split
from .model import HiddenErrorScale,ErrorScale,laplace_scale_loss
from .calibration import calibration_curve,matched_coverage_width


def compare_v2_scales(manifest,reference_run,model_directories,out,*,device='cpu',epochs=50):
    """Fit on scale, calibrate on cal, inspect val; never access old test labels."""
    df=load_manifest(manifest);lookup=dict(zip(df.song_id,df.file))
    split=_reference_split(df,reference_run)
    folders=[Path(p) for p in model_directories]
    models=[load_v2_checkpoint(p/'model.pt',device) for p in folders]
    with np.load(folders[0]/'normalizer.npz') as z:norm=(z['mean'],z['std'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Choose an empty v2 calibration output directory')
    cache={}
    for sid in split['scale']+split['cal']+split['val']:
        z=read_npz(lookup[sid]);p=predict_v2_ensemble(models,z,norm,device)
        cache[sid]=(z,p)
    def tensor(x):return torch.as_tensor(x,device=device).unsqueeze(0)
    def fit(mode):
        torch.manual_seed(2026)
        net=(HiddenErrorScale(models[0].hidden_dim) if mode=='hidden'
             else ErrorScale(models[0].hidden_dim)).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for epoch in range(epochs):
            order=np.random.default_rng(2026+epoch).permutation(split['scale'])
            net.train()
            for sid in order:
                z,p=cache[sid]
                opt.zero_grad(set_to_none=True)
                if mode=='hidden':scale=net(tensor(p['hidden']))
                else:scale=net(tensor(p['hidden']),tensor(p['tau']),
                    tensor(p['std_model']),tensor(p['mean']),tensor(z['times']),
                    step_seconds=nominal_step(z))
                loss=laplace_scale_loss(scale,tensor(p['mean']),
                    tensor(np.nan_to_num(z['mean'],nan=0)),tensor(z['mask']).bool())
                loss.backward();opt.step()
        net.eval()
        torch.save({'mode':mode,'hidden':models[0].hidden_dim,
                    'state_dict':net.state_dict()},out/f'{mode}_scale.pt')
        return net
    nets={'uniform':None,'hidden':fit('hidden'),'transition':fit('transition')}
    def make_item(sid,net,mode):
        z,p=cache[sid]
        with torch.inference_mode():
            if mode=='uniform':scale=np.ones_like(p['mean'])
            elif mode=='hidden':scale=net(tensor(p['hidden']))[0].cpu().numpy()
            else:scale=net(tensor(p['hidden']),tensor(p['tau']),
                tensor(p['std_model']),tensor(p['mean']),tensor(z['times']),
                step_seconds=nominal_step(z))[0].cpu().numpy()
        return {'song_id':sid,'pred':p['mean'],'ref':z['mean'],
                'mask':z['mask'],'scale':scale}
    curves={}
    for mode,net in nets.items():
        cal=[make_item(sid,net,mode) for sid in split['cal']]
        val=[make_item(sid,net,mode) for sid in split['val']]
        curves[mode]=calibration_curve(cal,val,nominals=(.7,.8,.85,.9,.95))
    pd.DataFrame([{'method':mode,**row} for mode,curve in curves.items()
                  for row in curve]).to_csv(out/'coverage_width.csv',index=False)
    default90=next(row for row in curves['uniform'] if np.isclose(row['nominal'],.9))
    (out/'default_uniform_calibration.json').write_text(json.dumps({
        'model_version':'v2','method':'uniform','nominal':.9,
        'q':default90['q'],'n_cal':default90['n_cal'],
        'model_directories':[str(p.resolve()) for p in folders],
        'note':'Calibrated on saved DEAM calibration songs; old test was not evaluated.'},indent=2))
    comparisons=[]
    for mode in ('hidden','transition'):
        comparisons.extend([{'comparison':f'{mode}_vs_uniform',**row}
             for row in matched_coverage_width(curves[mode],curves['uniform'],
                                               (.7,.75,.8,.85,.9,.95))])
    pd.DataFrame(comparisons).to_csv(out/'matched_empirical_coverage.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({
        'default':'uniform','optional':['hidden','transition'],
        'scale_songs':len(split['scale']),'calibration_songs':len(split['cal']),
        'evaluation_songs':len(split['val']),'test_songs_used':False,
        'caution':'Validation songs were used for point-model selection; their interval coverage is exploratory and is not a confirmatory guarantee.'},indent=2))
    return curves
