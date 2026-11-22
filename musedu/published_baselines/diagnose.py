"""Standalone Librosa/SciPy/Numba smoke probe without reading any DEAM labels.

Example:
 python -m musedu.published_baselines.diagnose --cache-dir C:\\Users\\you\\numba_cache
 python -m musedu.published_baselines.diagnose --audio path/to/a/song.mp3 --seconds 4
"""
from __future__ import annotations
import argparse
import faulthandler
import os
from pathlib import Path
import time


def log(message):
    print(f"[doctor {time.strftime('%H:%M:%S')}] {message}",flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description='Isolate slow Librosa initialization and audio features')
    ap.add_argument('--audio',default=None,help='Optional audio path; otherwise use synthetic 220-Hz tone')
    ap.add_argument('--seconds',type=float,default=3.,help='Clip length used for diagnosis')
    ap.add_argument('--cache-dir',default=None,help='Writable NUMBA_CACHE_DIR; set before importing librosa')
    ap.add_argument('--trace-after',type=int,default=45)
    ap.add_argument('--decoder',choices=['soundfile','librosa'],default='soundfile')
    a=ap.parse_args(argv)
    if not (0.5<=a.seconds<=30.): ap.error('--seconds must be between 0.5 and 30')
    if a.cache_dir:
        d=Path(a.cache_dir).expanduser().resolve()
        d.mkdir(parents=True,exist_ok=True)
        p=d/'write_test.tmp';p.write_text('ok');p.unlink()
        os.environ['NUMBA_CACHE_DIR']=str(d)
    log('NUMBA_CACHE_DIR='+str(os.environ.get('NUMBA_CACHE_DIR','not set')))
    for mod in ('numpy','scipy','numba','soundfile','soxr','librosa'):
        log('START import '+mod)
        t=time.perf_counter()
        try:
            x=__import__(mod)
            log('DONE import '+mod+' '+str(getattr(x,'__version__',''))+' in '+f'{time.perf_counter()-t:.3f}s')
        except Exception as e:
            log(f'FAILED import {mod}: {type(e).__name__}: {e}')
            raise
    if a.trace_after:
        faulthandler.enable()
        faulthandler.dump_traceback_later(a.trace_after,repeat=True)
    import numpy as np
    from .features import frame_features_librosa,decode_audio
    try:
        if a.audio:
            log('START decode '+a.audio)
            audio,sr=decode_audio(a.audio,decoder=a.decoder)
            audio=audio[:int(a.seconds*sr)]
            log('DONE decode '+str(len(audio))+' samples')
        else:
            sr=44100
            samples=np.arange(int(sr*a.seconds))/sr
            audio=(0.1*np.sin(2*np.pi*220*samples)).astype(np.float32)
        times=np.arange(.5,len(audio)/sr+.000001,.5,dtype=np.float32)
        log(f'START Librosa 77-D features, audio seconds={len(audio)/sr:.2f}, times={len(times)}')
        t=time.perf_counter()
        features=frame_features_librosa(audio,sr,times)
        log(f'DONE features: {features.shape}, finite={bool(np.isfinite(features).all())}, elapsed={time.perf_counter()-t:.2f}s')
        if features.shape!=(len(times),77) or not np.isfinite(features).all():
            raise AssertionError('77-D feature computation produced bad shape or non-finite values')
        log('PASS: no DEAM training performed, no benchmark results produced')
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
