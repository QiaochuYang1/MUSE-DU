"""Provenance-aware adapters for DEAM/PMEmo averaged CSV and MERP rater Parquet.

No dataset files are redistributed. Each adapter fails explicitly on ambiguous schemas.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

AXES = ("valence", "arousal")

@dataclass
class Annotation:
    dataset: str
    song_id: str
    times: np.ndarray                 # seconds, absolute audio time
    mean: np.ndarray                  # [T,2], target convention [-1,1]
    std: np.ndarray                   # [T,2], nan when unavailable
    mask: np.ndarray                  # [T,2] boolean
    std_mask: np.ndarray              # [T,2] boolean
    raters: np.ndarray                # [T,2] count or nan

    def validate(self):
        t = len(self.times)
        if t < 1 or not np.all(np.diff(self.times) > 0):
            raise ValueError(f"{self.song_id}: nonincreasing or empty timestamps")
        for name in ("mean", "std", "mask", "std_mask", "raters"):
            if getattr(self, name).shape != (t, 2):
                raise ValueError(f"{self.song_id}: invalid {name} shape")
        if not np.any(self.mask):
            raise ValueError(f"{self.song_id}: no valid labels")
        if np.any(np.abs(self.mean[self.mask]) > 1.0001):
            raise ValueError(f"{self.song_id}: label outside [-1,1]; check --label-scale")
        if np.any(self.std[self.std_mask] < 0):
            raise ValueError(f"{self.song_id}: negative disagreement")
        if np.any(self.std_mask & ~self.mask):
            raise ValueError(f"{self.song_id}: dispersion without a mean label")
        return self


def _scale(values, mode):
    x = np.asarray(values, dtype=float)
    if mode == 'minus_one_one':
        pass
    elif mode == 'zero_one':
        x = 2*x - 1
    elif mode == 'one_nine':
        x = (x-5)/4
    else:
        raise ValueError('Explicit scale required: minus_one_one / zero_one / one_nine')
    return x


def _std_scale(values, mode):
    x = np.asarray(values, dtype=float)
    return x * ({'minus_one_one': 1., 'zero_one': 2., 'one_nine': .25}[mode])


def _time_columns(df):
    """Wide CSV: identifier then time columns sample_1... or 15.0 / 15.5 etc.
    Explicit time_<seconds> columns override index-based time positioning.
    """
    candidates = []
    for c in df.columns:
        s = str(c).strip()
        m = re.fullmatch(r'(?:sample|frame|step|t)[_ ]?(\d+)',s,re.I)
        if m:
            candidates.append((c, float(m.group(1)), 'indexed'))
            continue
        m = re.fullmatch(r'(?:time|sec|seconds)[_ ]?(\d+(?:\.\d+)?)',s,re.I)
        if m:
            candidates.append((c, float(m.group(1)), 'absolute'))
            continue
        if re.fullmatch(r'\d+(?:\.\d+)?',s):
            candidates.append((c, float(s), 'numeric'))
    if len(candidates) < 2:
        raise ValueError('No time columns identified (expected sample_1... or time_15.0...)')
    kinds = {c[2] for c in candidates}
    if len(kinds) != 1:
        raise ValueError('Mixed time-column conventions: convert annotations explicitly')
    candidates.sort(key=lambda v:v[1])
    return candidates


def read_wide_table(path: Path, *, fps:float, start:float, scale:str, is_std=False):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f'{path}: empty CSV')
    cols = _time_columns(df)
    id_name = next((x for x in df.columns if str(x).strip().lower() in ('song_id','musicid','music_id','songid','id')),None)
    if id_name is None:
        non_times=[x for x in df.columns if x not in {v[0] for v in cols}]
        if len(non_times)!=1:
            raise ValueError(f'{path}: ambiguous song identifier; expected song_id/id')
        id_name=non_times[0]
    kind=cols[0][2]
    indices=np.asarray([v[1] for v in cols],float)
    if kind=='indexed':
        # Explicit assumption: sample_1 denotes first retained label bin (start seconds).
        ts=start + (indices-indices[0])/fps
    elif kind=='absolute':
        ts=indices
    else:
        # Plain numeric columns are ambiguous: require explicit time_/sample_ names.
        raise ValueError(f'{path}: numeric-only time headers ambiguous; use time_<seconds> or sample_<index>')
    out={}
    for _,row in df.iterrows():
        sid=str(row[id_name]).strip()
        if sid.endswith('.0') and sid[:-2].isdigit():sid=sid[:-2]
        if sid in out:raise ValueError(f'{path}: duplicate song {sid}')
        values=pd.to_numeric(row[[v[0] for v in cols]], errors='coerce').to_numpy(dtype=float)
        vals=_std_scale(values,scale) if is_std else _scale(values,scale)
        out[sid]=(ts.copy(),vals)
    return out


def _choose_file(root, axis, stat, explicit=None):
    if explicit:
        p=Path(explicit)
        if not p.is_file():raise FileNotFoundError(p)
        return p
    # Matches DEAM/PMEmo official format arousal_cont_average.csv,
    # and '.../arousal.csv' in a dynamic mean subdirectory.
    options=[]
    for p in root.rglob('*.csv'):
        name=p.name.lower()
        parent=str(p.parent).lower()
        if axis not in name:continue
        if any(x in name for x in ('static','song_level')):continue
        if stat=='mean' and ('average' in name or 'mean' in name or (('cont' in parent or 'dynamic' in parent) and 'std' not in name and 'sd' not in name)):
            if 'std' not in name:options.append(p)
        if stat=='std' and ('std' in name or 'standard_deviation' in name):options.append(p)
    if len(options)!=1:
        raise ValueError(f'{root}: {axis} {stat}: found {len(options)} matching CSVs: '
                         f'{[str(v.relative_to(root)) for v in options[:12]]}; '
                         'select exact path via --valence-file / --arousal-file / --valence-std-file / --arousal-std-file')
    return options[0]


def load_averaged_csv(root, dataset, fps=2., start=15., scale='minus_one_one',
                     valence_file=None, arousal_file=None, valence_std_file=None, arousal_std_file=None,
                     require_std=False):
    """Read official-style DEAM or PMEmo wide dynamic annotation CSVs.

    Never auto-unzip or auto-download copyrighted audio. Multiple DEAM releases should
    be consolidated or passed as a single compatible annotation set explicitly.
    """
    root=Path(root)
    if dataset not in ('deam','pmemo'):raise ValueError(dataset)
    if dataset=='deam' and (root/'DEAM_Annotations').is_dir() and not any(
            (valence_file,arousal_file,valence_std_file,arousal_std_file)):
        return load_deam_release(root,scale=scale)
    if dataset=='pmemo' and (root/'dynamic_annotations.csv').is_file() and not any(
            (valence_file,arousal_file,valence_std_file,arousal_std_file)):
        return load_pmemo_long(root,scale=scale)
    explicit=[valence_file,arousal_file]
    mean=[];std=[]
    for a,e in zip(AXES,explicit):
        mean.append(read_wide_table(_choose_file(root,a,'mean',e),fps=fps,start=start,scale=scale))
    for a,e in zip(AXES,[valence_std_file,arousal_std_file]):
        try:
            p=_choose_file(root,a,'std',e)
            std.append(read_wide_table(p,fps=fps,start=start,scale=scale,is_std=True))
        except ValueError:
            if require_std or e:raise
            std.append({})
    items=[]
    for sid in sorted(set(mean[0]) & set(mean[1])):
        tv,v=mean[0][sid];ta,a=mean[1][sid]
        if not np.allclose(tv,ta): raise ValueError(f'{sid}: valence/arousal times differ')
        y=np.column_stack([v,a]); s=np.full_like(y,np.nan)
        for d in range(2):
            if sid in std[d]:
                st,sd=std[d][sid]
                if not np.allclose(tv,st):raise ValueError(f'{sid}: std time mismatch')
                s[:,d]=sd
        mask=np.isfinite(y); dm=np.isfinite(s)&(s>=0)&mask
        items.append(Annotation(dataset,sid,tv,y,s,mask,dm,np.full_like(y,np.nan)).validate())
    if not items:raise ValueError('No shared song IDs between dynamic valence and arousal')
    return items


def load_deam_release(root, *, scale='minus_one_one'):
    """DEAM release: absolute millisecond columns and separate per-rater tables."""
    if scale!='minus_one_one':
        raise ValueError('DEAM release labels are already in [-1,1]')
    base=Path(root)/'DEAM_Annotations'/'annotations'
    avg=base/'annotations averaged per song'/'dynamic (per second annotations)'
    individual=base/'annotations per each rater'/'dynamic (per second annotations)'
    tables={axis:pd.read_csv(avg/f'{axis}.csv').set_index('song_id') for axis in AXES}
    if any(t.index.has_duplicates for t in tables.values()):
        raise ValueError('DEAM duplicate song identifiers')
    items=[]
    for sid in sorted(set(tables['valence'].index)&set(tables['arousal'].index)):
        series={}
        for axis in AXES:
            row=tables[axis].loc[sid]
            series[axis]={float(m.group(1))/1000:float(value)
                for col,value in row.items()
                if (m:=re.fullmatch(r'sample_(\d+)ms',str(col))) and pd.notna(value)}
        times=np.array(sorted(set(series['valence'])|set(series['arousal'])),dtype=float)
        y=np.full((len(times),2),np.nan);std=np.full_like(y,np.nan)
        raters=np.full_like(y,np.nan)
        for j,axis in enumerate(AXES):
            y[:,j]=[series[axis].get(t,np.nan) for t in times]
            f=individual/axis/f'{sid}.csv'
            if not f.is_file():continue
            frame=pd.read_csv(f)
            for i,t in enumerate(times):
                col=f'sample_{int(round(t*1000))}ms'
                if col not in frame:continue
                vals=pd.to_numeric(frame[col],errors='coerce').dropna().to_numpy(dtype=float)
                raters[i,j]=len(vals)
                if len(vals)>=2:std[i,j]=np.std(vals,ddof=1)
        mask=np.isfinite(y);std_mask=np.isfinite(std)&mask
        items.append(Annotation('deam',str(sid),times,y,std,mask,std_mask,raters).validate())
    if not items:raise ValueError('DEAM release has no shared annotated songs')
    return items


def load_pmemo_long(root, *, scale='zero_one'):
    """PMEmo 2019 release: one row per song and 0.5-second audio timestamp."""
    root=Path(root)
    if scale!='zero_one':
        raise ValueError('PMEmo 2019 dynamic annotations use 0..1; pass --label-scale zero_one')
    mean=pd.read_csv(root/'dynamic_annotations.csv')
    required={'musicId','frameTime','Valence(mean)','Arousal(mean)'}
    if not required.issubset(mean):raise ValueError(f'PMEmo 2019 missing {required-set(mean)}')
    if mean.duplicated(['musicId','frameTime']).any():
        raise ValueError('PMEmo 2019 repeated song/time annotation')
    std_file=root/'dynamic_annotations_std.csv'
    if std_file.is_file():
        std=pd.read_csv(std_file)
        expected={'musicId','frameTime','Valence(std)','Arousal(std)'}
        if not expected.issubset(std):raise ValueError(f'PMEmo 2019 missing {expected-set(std)}')
        if std.duplicated(['musicId','frameTime']).any():
            raise ValueError('PMEmo 2019 repeated song/time dispersion')
        mean=mean.merge(std[list(expected)],on=['musicId','frameTime'],how='left',validate='one_to_one')
    else:
        mean['Valence(std)']=np.nan;mean['Arousal(std)']=np.nan
    items=[]
    for sid,group in mean.groupby('musicId',sort=True):
        group=group.sort_values('frameTime')
        times=group['frameTime'].to_numpy(dtype=float)
        if len(times)>1 and not np.allclose(np.diff(times),.5):
            raise ValueError(f'PMEmo {sid}: unexpected annotation step')
        y=_scale(group[['Valence(mean)','Arousal(mean)']].to_numpy(dtype=float),scale)
        s=_std_scale(group[['Valence(std)','Arousal(std)']].to_numpy(dtype=float),scale)
        mask=np.isfinite(y);std_mask=np.isfinite(s)&(s>=0)&mask
        items.append(Annotation('pmemo',str(sid),times,y,s,mask,std_mask,
                                np.full_like(y,np.nan)).validate())
    if not items:raise ValueError('PMEmo 2019 has no valid dynamic annotations')
    return items


def load_merp(root, *, filtered=True, fps=2.):
    """Rater-level MERP arrays in the official Parquet structure.

    The released filtered arrays are 2 Hz and start 15 seconds into the audio;
    the raw arrays are 10 Hz and start at audio time zero. These grids are
    verified against the released WAV durations. Aggregation does not
    interpolate, and only valid raters contribute to each bin's dispersion.
    """
    root=Path(root)
    f=root/('annotations_filtered.parquet' if filtered else 'annotations_raw.parquet')
    if not f.is_file():raise FileNotFoundError(f)
    df=pd.read_parquet(f)
    expected={'song_id','rater_id','valence','arousal'}
    if not expected.issubset(df.columns):raise ValueError(f'{f}: missing {expected-set(df.columns)}')
    source_fps=2. if filtered else 10.
    start=15. if filtered else 0.
    if fps<=0 or source_fps/fps != round(source_fps/fps):
        raise ValueError(f'MERP output rate must divide {source_fps:g} Hz')
    down=int(round(source_fps/fps)); items=[]
    for sid,group in df.groupby('song_id'):
        if group.rater_id.duplicated().any():
            raise ValueError(f'MERP {sid}: repeated rater/song entry (inspect sessions first)')
        series=[]
        for row in group.itertuples(index=False):
            v=np.asarray(row.valence,dtype=float);a=np.asarray(row.arousal,dtype=float)
            if len(v)==0 or len(a)==0:continue
            n=min(len(v),len(a)); n=n//down
            if n==0:continue
            x=np.column_stack([v[:n*down],a[:n*down]]).reshape(n,down,2)
            # Do not turn fully missing windows into zeros.
            valid=np.isfinite(x)
            counts=valid.sum(axis=1)
            obs=np.divide(np.where(valid,x,0).sum(axis=1),counts,
                          out=np.full((n,2),np.nan),where=counts>0)
            if np.any(np.isfinite(obs) & (np.abs(obs)>1.0001)):
                raise ValueError(f'MERP {sid}: labels outside [-1,1]')
            series.append(obs)
        if not series:continue
        length=max(len(v) for v in series)
        arr=np.full((len(series),length,2),np.nan,dtype=float)
        for k,v in enumerate(series):arr[k,:len(v)]=v
        n=np.isfinite(arr).sum(0)
        mu=np.divide(np.nansum(arr,axis=0),n,out=np.full((length,2),np.nan),where=n>0)
        diff=np.where(np.isfinite(arr),arr-mu[None],0)
        st=np.sqrt(np.divide(np.sum(diff**2,axis=0),n-1,
                             out=np.full((length,2),np.nan),where=n>=2))
        ts=start+np.arange(length,dtype=float)/fps
        items.append(Annotation('merp',str(sid),ts,mu,st,n>0,n>=2,n.astype(float)).validate())
    if not items:raise ValueError('MERP has no valid song annotations')
    return sorted(items,key=lambda x:x.song_id)


def find_audio(root: Path, song_id:str):
    """Exact stem match prevents accidental 1/11 ID collisions."""
    matches=[]
    for suffix in ('.mp3','.wav','.flac','.ogg'):
        matches+=list(Path(root).rglob(f'{song_id}{suffix}'))
    if len(matches)!=1:raise FileNotFoundError(f'{song_id}: expected exactly one audio match; got {len(matches)}')
    return matches[0]
