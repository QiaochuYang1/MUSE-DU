"""Leakage-guarded DEAM/PMEmo/MERP baseline training and zero-shot transfer.

Fitted baselines are *adaptations to the shared protocol*. No original-paper
scores are embedded, and zero-shot external labels are never used for selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .baselines import METHODS, NEURAL_METHODS, CLASSICAL_METHODS, BaselineNetwork, choose_view, frame_regression_loss
from .experiment import (load_manifest, read_npz, split_songs, provenance, nominal_step, pack,
                         seed_everything, on_device)
from .calibration import song_score, conformal_quantile, evaluate_intervals
from .metrics import song_metrics, aggregate


def _index(df):
    return dict(zip(df.song_id.astype(str), df.file))


def _load_split(df, seed, reference_run=None):
    if reference_run is None:
        return split_songs(df, seed=seed)
    parent = Path(reference_run)
    splits = json.loads((parent/'splits.json').read_text())
    if set(splits) != {'train', 'val', 'scale', 'cal', 'test'}:
        raise ValueError('Reference experiment must have five named song splits')
    all_ids = [str(sid) for ids in splits.values() for sid in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(df.song_id):
        raise ValueError('Reference splits do not match this dataset manifest or contain overlapping songs')
    old = json.loads((parent/'source_manifest.json').read_text())
    original = {(str(r['dataset']), str(r['song_id'])):r['audio_sha256'] for r in old}
    if original != {(r['dataset'],r['song_id']):r['audio_sha256'] for r in provenance(df)}:
        raise ValueError('Reference run has different source audio or song IDs')
    return splits


def _validate_source(df):
    if df.dataset.nunique() != 1:
        raise ValueError('A benchmark must develop on exactly one corpus')
    src = provenance(df)
    hashes = [r['audio_sha256'] for r in src]
    if any(not x for x in hashes):
        raise ValueError('Missing decoded-audio fingerprints; rerun `musedu prepare`')
    if len(hashes) != len(set(hashes)):
        raise ValueError('Duplicate decoded audio: fix overlap before splitting')
    return src


def _validate_view(df, view):
    dims = set()
    for f in df.file:
        z = read_npz(f)
        x = choose_view(z, view)
        if not np.isfinite(x).all():
            raise ValueError(f'{f}: nonfinite features')
        if len(x) != len(z['times']):
            raise ValueError(f'{f}: feature/timing misalignment')
        dims.add(x.shape[1])
    if len(dims) != 1:
        raise ValueError('Mixed feature dimensions or acoustic/MERT backends within manifest')
    return dims.pop()


def train_normalizer(df, ids, view):
    lookup = _index(df)
    count=0;total=None;total2=None
    for sid in ids:
        x = choose_view(read_npz(lookup[sid]),view).astype('float64')
        if total is None:
            total = np.zeros(x.shape[1],dtype='float64')
            total2 = np.zeros(x.shape[1],dtype='float64')
        total += x.sum(0); total2 += (x*x).sum(0);count += len(x)
    if count==0:raise ValueError('Empty training features')
    mean = total/count
    std = np.sqrt(np.maximum(total2/count - mean**2, 0)).clip(min=1e-4)
    return mean.astype('float32'),std.astype('float32')


class BaselineWindows(Dataset):
    def __init__(self, df, ids, normalizer, view, window=120):
        if window<2:raise ValueError('Window must have at least two bins')
        lookup = _index(df)
        self.windows=[]
        mean, std = normalizer
        for sid in ids:
            z = read_npz(lookup[sid]); x=(choose_view(z,view)-mean)/std
            for start in range(0,len(x),window):
                stop=min(start+window,len(x))
                if not z['mask'][start:stop].any():continue
                self.windows.append({'x':x[start:stop],
                                     'times':z['times'][start:stop],
                                     'mean':z['mean'][start:stop],
                                     'std':z['std'][start:stop],
                                     'mask':z['mask'][start:stop],
                                     'std_mask':z['std_mask'][start:stop],
                                     'step_seconds':nominal_step(z)})
        if not self.windows:raise ValueError('No labeled windows')

    def __len__(self):return len(self.windows)
    def __getitem__(self,j):return self.windows[j]


def _new_model(method, in_features, hidden):
    return BaselineNetwork(method, in_features, hidden=hidden)


def _train_neural(df, split, norm, view, method, out, *, hidden, window, epochs, lr, batch_size, seed, device):
    seed_everything(seed)
    train=BaselineWindows(df,split['train'],norm,view,window)
    val=BaselineWindows(df,split['val'],norm,view,window)
    tr=DataLoader(train,batch_size=batch_size,shuffle=True,collate_fn=pack)
    va=DataLoader(val,batch_size=batch_size,shuffle=False,collate_fn=pack)
    model=_new_model(method,len(norm[0]),hidden).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    best=float('inf');bad=0
    for _ in range(epochs):
        model.train()
        for batch in tr:
            z=on_device(batch,device)
            optim.zero_grad(set_to_none=True)
            pred=model(z['x'],z['times'],z['valid'])
            loss=frame_regression_loss(pred,z['mean'],z['mask'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        model.eval();errors=[]
        with torch.inference_mode():
            for batch in va:
                z=on_device(batch,device)
                errors.append(float(frame_regression_loss(model(z['x'],z['times'],z['valid']),
                                                         z['mean'],z['mask'])))
        value=float(np.mean(errors))
        if value<best-1e-7:
            bad=0;best=value
            torch.save({'state_dict':model.state_dict(), 'method':method,
                        'in_features':len(norm[0]),'hidden':hidden},out/'baseline.pt')
        else:
            bad+=1
            if bad>=7:break
    return best


def _make_estimator(method, seed):
    if method=='ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.)
    if method=='random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=180,min_samples_leaf=3,
                                     random_state=seed,n_jobs=2)
    if method=='xgboost':
        try:from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError('Install XGBoost using pip install -e ".[baselines]"') from exc
        return XGBRegressor(n_estimators=250,max_depth=5,learning_rate=.05,
                            subsample=.85,colsample_bytree=.85,
                            objective='reg:squarederror',tree_method='hist',
                            n_jobs=2,random_state=seed)
    raise ValueError(method)


def _train_classical(df,split,norm,view,method,out,*,seed):
    import joblib
    lookup=_index(df);mu,sd=norm
    models=[]
    # Per-axis masks can differ: NEVER fill missing emotion annotations with zero targets.
    for d in range(2):
        all_x=[];all_y=[];all_w=[]
        for sid in split['train']:
            z=read_npz(lookup[sid]);mask=z['mask'][:,d].astype(bool)
            if not mask.any():continue
            xx=(choose_view(z,view)[mask]-mu)/sd
            all_x.append(xx)
            all_y.append(z['mean'][mask,d])
            all_w.append(np.full(mask.sum(),1./mask.sum(),dtype='float64'))
        if not all_x:raise ValueError(f'No valid axis {d} labels on train set')
        x=np.concatenate(all_x);y=np.concatenate(all_y)
        w=np.concatenate(all_w);w /= w.sum()
        estimator=_make_estimator(method,seed+d)
        estimator.fit(x,y,sample_weight=w*len(x))
        models.append(estimator)
    joblib.dump(models,out/'baseline.joblib')
    return None


def _load_predictor(folder,device):
    meta=json.loads((folder/'baseline_metadata.json').read_text())
    if meta['method'] in CLASSICAL_METHODS:
        import joblib
        return joblib.load(folder/'baseline.joblib'),meta
    state=torch.load(folder/'baseline.pt',weights_only=True,map_location=device)
    model=_new_model(state['method'],state['in_features'],state['hidden']).to(device)
    model.load_state_dict(state['state_dict']);model.eval()
    return model,meta


def _predict(predictor, meta, z, norm, device='cpu', window=120):
    mu,sd=norm
    x=(choose_view(z,meta['view'])-mu)/sd
    if isinstance(predictor,list):
        return np.clip(np.column_stack([m.predict(x) for m in predictor]),-1,1).astype('float32')
    preds=[]
    with torch.inference_mode():
        for start in range(0,len(x),window):
            stop=min(len(x),start+window)
            inp=torch.as_tensor(x[start:stop][None],dtype=torch.float32,device=device)
            tm=torch.as_tensor(z['times'][start:stop][None],dtype=torch.float32,device=device)
            valid=torch.ones((1,stop-start),dtype=torch.bool,device=device)
            preds.append(predictor(inp,tm,valid)[0].cpu().numpy())
    return np.concatenate(preds).astype('float32')


def _score(df,ids,predictor,meta,norm,device,window):
    lookup=_index(df);metrics=[];scores=[];items=[];rows=[]
    for sid in ids:
        z=read_npz(lookup[sid]);p=_predict(predictor,meta,z,norm,device,window)
        scale=np.ones_like(p,dtype='float32')
        score=song_score(p,z['mean'],scale,z['mask'])
        record=song_metrics(p,z['mean'],z['mask'])
        metrics.append(record);scores.append(score)
        rows.append({'song_id':str(sid),**record})
        items.append({'song_id':sid,'pred':p,'ref':z['mean'],'mask':z['mask'],'scale':scale})
    return scores,metrics,items,rows


def _normalizer(folder):
    with np.load(folder/'normalizer.npz') as z:
        return z['mean'],z['std']


def run_baseline_training(manifest,out,method,*,reference_run=None,device='cpu',epochs=25,
                          seed=2026,hidden=96,window=120,lr=3e-4,batch_size=8):
    if method not in METHODS: raise ValueError(f'Unknown method {method}; select from {sorted(METHODS)}')
    if epochs<1 or window<2: raise ValueError('epochs>=1 and window>=2 required')
    df=load_manifest(manifest)
    sources=_validate_source(df)
    view=METHODS[method]
    dimension=_validate_view(df,view)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a clean baseline output directory; refusing to overwrite experimental artifacts')
    splits=_load_split(df,seed,reference_run)
    norm=train_normalizer(df,splits['train'],view)
    np.savez(out/'normalizer.npz',mean=norm[0],std=norm[1])
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    (out/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    meta={'method':method,'view':view,'feature_dim':dimension,'development_dataset':str(df.dataset.iloc[0]),
          'training_seed':seed,'window':window,'hidden':hidden,
          'note':'Retrained protocol-matched baseline, NOT published author weights or scores'}
    (out/'baseline_metadata.json').write_text(json.dumps(meta,indent=2))
    if method in CLASSICAL_METHODS:
        _train_classical(df,splits,norm,view,method,out,seed=seed)
    else:
        _train_neural(df,splits,norm,view,method,out,hidden=hidden,window=window,
                      epochs=epochs,lr=lr,batch_size=batch_size,seed=seed,device=device)
    predictor,meta=_load_predictor(out,device)
    cs,_,_,_=_score(df,splits['cal'],predictor,meta,norm,device,window)
    q=conformal_quantile(cs,alpha=.1)
    (out/'calibration.json').write_text(json.dumps({'q':q,'alpha':.1,'n_cal':len(cs),
                'score':'simultaneous max normalized by uniform scale 1'},indent=2))
    _,reports,items,rows=_score(df,splits['test'],predictor,meta,norm,device,window)
    from pandas import DataFrame
    DataFrame(rows).to_csv(out/'test_song_metrics.csv',index=False)
    report={'method':method,'input_view':view,'split_sizes':{k:len(v) for k,v in splits.items()},
            'point_metrics':aggregate(reports),'uniform_conformal':evaluate_intervals(items,q),
            'results_origin':'trained locally on the supplied development manifest'}
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2))
    return report


def run_baseline_external(manifest,trained,device='cpu',window=None):
    df=load_manifest(manifest)
    folder=Path(trained)
    source=json.loads((folder/'source_manifest.json').read_text())
    if df.dataset.nunique()!=1:raise ValueError('External manifest must contain exactly one dataset')
    if str(df.dataset.iloc[0]) in {r['dataset'] for r in source}:
        raise ValueError('Cannot evaluate development corpus as external; use its held-out test split')
    skipped=[]
    if str(df.dataset.iloc[0])=='merp':
        skipped=df[df.song_id.str.startswith('deam_')].song_id.tolist()
        df=df[~df.song_id.str.startswith('deam_')].copy()
        if df.empty:raise ValueError('Only known DEAM anchor songs remain in MERP')
    external=provenance(df)
    if any(not r['audio_sha256'] for r in source+external):
        raise ValueError('Missing decoded audio SHA256; regenerate cached features')
    if {r['audio_sha256'] for r in source}&{r['audio_sha256'] for r in external}:
        raise ValueError('Audio overlap between development and external datasets')
    predictor,meta=_load_predictor(folder,device)
    _validate_view(df,meta['view'])
    norm=_normalizer(folder)
    if len(norm[0]) != _validate_view(df,meta['view']):
        raise ValueError('External embedding dimension does not match training; check feature backend/model version')
    if window is None:window=meta['window']
    _,reports,items,rows=_score(df,df.song_id.tolist(),predictor,meta,norm,device,window)
    q=json.loads((folder/'calibration.json').read_text())['q']
    result={'method':meta['method'],'external_dataset':str(df.dataset.iloc[0]),
            'excluded_known_deam_anchors':skipped,'point_metrics':aggregate(reports),
            'uniform_conformal':evaluate_intervals(items,q),
            'caution':'Zero-shot external interval coverage is empirical; source-corpus calibration guarantees do not transfer across domain shift'}
    from pandas import DataFrame
    DataFrame(rows).to_csv(folder/f'external_{df.dataset.iloc[0]}_song_metrics.csv',index=False)
    (folder/f'external_{df.dataset.iloc[0]}_metrics.json').write_text(json.dumps(result,indent=2))
    return result
