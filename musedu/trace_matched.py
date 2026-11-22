"""Run the prespecified T1/T2 TRACE-MER matched ablations."""
from __future__ import annotations
import argparse
from pathlib import Path
from .valence_experiment import run


def main(manifest,reference,out,device='cuda',epochs=15):
    root=Path(out)
    for label,architecture in (("T1_no_cue_sufficiency","trace_no_cue_loss"),
                               ("T2_no_cues","trace_no_cues")):
        for seed in range(2026,2031):
            run(manifest,reference,root/label/f'seed{seed}',architecture=architecture,
                representation='mix',delay=1.,harmonic=True,seed=seed,
                epochs=epochs,device=device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=15)
    a=p.parse_args();main(a.manifest,a.reference,a.out,a.device,a.epochs)
