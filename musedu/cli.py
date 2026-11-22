"""CLI: dataset audit, deterministic preprocessing, full training, external evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from .datasets import load_averaged_csv,load_merp,find_audio
from .features import prepare_one
from .experiment import run_training,run_external
from .baselines import METHODS
from .baseline_experiment import run_baseline_training,run_baseline_external
from .benchmark_suite import run_suite
from .v2_experiment import run_v2_development
from .v2_calibration import compare_v2_scales


def ann_from_args(a):
    if a.dataset=='merp':return load_merp(a.root,filtered=not a.raw_merp,fps=a.fps)
    return load_averaged_csv(a.root,a.dataset,fps=a.fps,start=a.start,
                 scale=a.label_scale, valence_file=a.valence_file,
                 arousal_file=a.arousal_file,valence_std_file=a.valence_std_file,
                 arousal_std_file=a.arousal_std_file,require_std=a.require_std)


def audit(args,prepare=False):
    anns=ann_from_args(args)
    audio_root=Path(args.audio_root or args.root)
    missing=[];good=[];rows=[]
    for ann in anns:
        try:
            audio=find_audio(audio_root,ann.song_id)
            good.append((ann,audio))
        except FileNotFoundError:
            missing.append(ann.song_id)
    report={'dataset':args.dataset,'annotations':len(anns),'audio_matched':len(good),
            'audio_missing':missing,'dispersion_songs':sum(int(a.std_mask.any()) for a in anns),
            'annotation_bins':sum(len(a.times) for a in anns),
            'source':str(Path(args.root).resolve()),'label_scale':
            ('minus_one_one' if args.dataset=='merp' else args.label_scale),
            'note':'File-level availability only; audio decoding is verified during prepare.'}
    print(json.dumps(report,indent=2))
    if prepare:
        if missing and args.strict:
            raise FileNotFoundError(f'Missing audio for {len(missing)} song(s): {missing[:15]}')
        folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
        failures=[]
        for ann,audio in good:
            dest=folder/'features'/args.dataset/f'{ann.song_id}.npz'
            try:
                n=prepare_one(ann,audio,dest,backend=args.backend,fps=args.fps,device=args.device)
                rows.append({'dataset':args.dataset,'song_id':ann.song_id,'file':str(dest.resolve()),
                             'audio':str(audio.resolve()),'n_bins':n})
            except Exception as exc:
                failures.append({'song_id':ann.song_id,'error':str(exc)})
        pd.DataFrame(rows,columns=['dataset','song_id','file','audio','n_bins']).to_csv(
            folder/f'{args.dataset}_manifest.csv',index=False)
        report.update({'prepared':len(rows),'failures':failures})
        (folder/f'{args.dataset}_audit.json').write_text(json.dumps(report,indent=2))
        print(f'Prepared {len(rows)} songs, failed {len(failures)}; manifest under {folder}')
        if failures and args.strict:raise RuntimeError('Some audio/feature preparation failed (see audit report)')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(prog='musedu',description='MUSE-DU reproducible research pipeline')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('audit','prepare'):
        x=sub.add_parser(cmd)
        x.add_argument('--dataset',required=True,choices=['deam','pmemo','merp'])
        x.add_argument('--root',required=True,help='Directory with annotation CSV or MERP Parquet')
        x.add_argument('--audio-root',default=None,help='Extracted audio directory')
        x.add_argument('--fps',type=float,default=2.)
        x.add_argument('--start',type=float,default=15.,help='First valid label time in seconds for indexed CSV columns')
        x.add_argument('--label-scale',default='minus_one_one',choices=['minus_one_one','zero_one','one_nine'])
        x.add_argument('--valence-file',default=None)
        x.add_argument('--arousal-file',default=None)
        x.add_argument('--valence-std-file',default=None)
        x.add_argument('--arousal-std-file',default=None)
        x.add_argument('--require-std',action='store_true')
        x.add_argument('--raw-merp',action='store_true')
        if cmd=='prepare':
            x.add_argument('--out',required=True)
            x.add_argument('--backend',choices=['acoustic','mert'],default='acoustic')
            x.add_argument('--device',default='cpu')
            x.add_argument('--strict',action='store_true',default=True)
            x.add_argument('--allow-missing',action='store_false',dest='strict')
    tr=sub.add_parser('train')
    tr.add_argument('--manifest',required=True)
    tr.add_argument('--out',required=True)
    tr.add_argument('--device',default='cpu')
    tr.add_argument('--ensemble',type=int,default=5)
    tr.add_argument('--epochs',type=int,default=25)
    tr.add_argument('--scale-epochs',type=int,default=50)
    tr.add_argument('--window',type=int,default=120)
    tr.add_argument('--hidden',type=int,default=96)
    tr.add_argument('--seed',type=int,default=2026)
    v2=sub.add_parser('v2-develop',help='Train a v2 prototype using reference train/val songs only')
    v2.add_argument('--manifest',required=True)
    v2.add_argument('--reference-run',required=True)
    v2.add_argument('--out',required=True)
    v2.add_argument('--device',default='cpu')
    v2.add_argument('--epochs',type=int,default=15)
    v2.add_argument('--seed',type=int,default=2026)
    v2.add_argument('--hidden',type=int,default=96)
    v2.add_argument('--batch-size',type=int,default=8)
    for key in ('point','song','shape','multi','dis'):
        v2.add_argument(f'--weight-{key}',type=float,default=None)
    v2cal=sub.add_parser('v2-calibrate',help='Compare v2 uniform and learned scales on development songs')
    v2cal.add_argument('--manifest',required=True)
    v2cal.add_argument('--reference-run',required=True)
    v2cal.add_argument('--model-dir',action='append',required=True)
    v2cal.add_argument('--out',required=True)
    v2cal.add_argument('--device',default='cpu')
    v2cal.add_argument('--epochs',type=int,default=50)
    ev=sub.add_parser('evaluate-external')
    ev.add_argument('--manifest',required=True)
    ev.add_argument('--trained',required=True)
    ev.add_argument('--device',default='cpu')
    ev.add_argument('--window',type=int,default=120)
    bs=sub.add_parser('baseline-train',help='Retrain a protocol-matched baseline on the same song splits')
    bs.add_argument('--manifest',required=True)
    bs.add_argument('--out',required=True)
    bs.add_argument('--method',required=True,choices=sorted(METHODS))
    bs.add_argument('--reference-run',default=None,help='MUSE-DU training directory with splits.json and source_manifest.json')
    bs.add_argument('--device',default='cpu')
    bs.add_argument('--epochs',type=int,default=25)
    bs.add_argument('--seed',type=int,default=2026)
    bs.add_argument('--hidden',type=int,default=96)
    bs.add_argument('--window',type=int,default=120)
    bs.add_argument('--lr',type=float,default=3e-4)
    bs.add_argument('--batch-size',type=int,default=8)
    be=sub.add_parser('baseline-external',help='Zero-shot evaluation of a frozen baseline on another corpus')
    be.add_argument('--manifest',required=True)
    be.add_argument('--trained',required=True)
    be.add_argument('--device',default='cpu')
    be.add_argument('--window',type=int,default=None)
    suite=sub.add_parser('baseline-suite',help='Train all selected baselines and export a unified CSV comparison')
    suite.add_argument('--manifest',required=True)
    suite.add_argument('--out',required=True)
    suite.add_argument('--reference-run',default=None)
    suite.add_argument('--methods',nargs='+',choices=sorted(METHODS),default=sorted(METHODS))
    suite.add_argument('--external-manifest',action='append',default=[],help='Repeat for PMEmo and MERP zero-shot tests')
    suite.add_argument('--device',default='cpu')
    suite.add_argument('--epochs',type=int,default=25)
    suite.add_argument('--seed',type=int,default=2026)
    suite.add_argument('--hidden',type=int,default=96)
    suite.add_argument('--window',type=int,default=120)
    suite.add_argument('--lr',type=float,default=3e-4)
    suite.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args(argv)
    if a.command in ('audit','prepare'):
        return audit(a,prepare=a.command=='prepare')
    if a.command=='baseline-suite':
        result=run_suite(a.manifest,a.out,methods=a.methods,reference_run=a.reference_run,
                         external_manifests=a.external_manifest,device=a.device,
                         epochs=a.epochs,seed=a.seed,hidden=a.hidden,
                         window=a.window,lr=a.lr,batch_size=a.batch_size)
    elif a.command=='baseline-train':
        result=run_baseline_training(a.manifest,a.out,a.method,reference_run=a.reference_run,
                            device=a.device,epochs=a.epochs,seed=a.seed,hidden=a.hidden,
                            window=a.window,lr=a.lr,batch_size=a.batch_size)
    elif a.command=='baseline-external':
        result=run_baseline_external(a.manifest,a.trained,device=a.device,window=a.window)
    elif a.command=='train':
        if a.ensemble<2: p.error('Ensemble must have >=2 members for variance estimation')
        result=run_training(a.manifest,a.out,device=a.device,ensemble=a.ensemble,
                            epochs=a.epochs,scale_epochs=a.scale_epochs,window=a.window,
                            hidden=a.hidden,seed=a.seed)
    elif a.command=='v2-develop':
        from .model import V2_LOSS_WEIGHTS
        weights={k:(getattr(a,f'weight_{k}') if getattr(a,f'weight_{k}') is not None
                    else V2_LOSS_WEIGHTS[k]) for k in V2_LOSS_WEIGHTS}
        result=run_v2_development(a.manifest,a.reference_run,a.out,weights=weights,
                    seed=a.seed,epochs=a.epochs,hidden=a.hidden,device=a.device,
                    batch_size=a.batch_size)
    elif a.command=='v2-calibrate':
        result=compare_v2_scales(a.manifest,a.reference_run,a.model_dir,a.out,
                                 device=a.device,epochs=a.epochs)
    else:result=run_external(a.manifest,a.trained,a.device,a.window)
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':main()
