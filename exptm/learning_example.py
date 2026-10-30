"""Optional same-example one-sample rerun, never part of the registered quantitative result."""
import argparse
from pathlib import Path
import numpy as np,torch
from run_checkpoint import setup
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=2026101304);a=p.parse_args();out=Path(a.output)
if out.exists():raise SystemExit('Choose a new output file; completed result is protected')
sampler,_,_=setup();torch.manual_seed(a.seed);x=sampler.sample_loop(num_samples=1,batch_size=1,temperature=5,pressure=0,eta=1,gamma=1).numpy();np.save(out,x);print('shape',x.shape,'density',(x[:,0]>=0).mean())
