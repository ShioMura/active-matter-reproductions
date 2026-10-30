"""Run the archived author-checkpoint example without changing original evidence."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--release-root', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('my_new_sample.npy'))
args = parser.parse_args()
if args.output.exists():
    raise SystemExit('Choose a new output filename.')
root = args.release_root.resolve()
checkpoint = root / 'author_source/GC_Lattice_gas/expTM_data_genenration/thermomaps-root/models/model_1.pt'
expected = '72125a26b5d136b8eeda782ff01fbecb99963b3ad58b8fce3f4494185f5419fc'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
sys.path.insert(0, str(root))
from run_checkpoint import setup

sampler, loader, metadata = setup()
print('Input shape:', tuple(loader.data.shape))
torch.manual_seed(2026101304)
samples = sampler.sample_loop(
    num_samples=1, batch_size=1,
    temperature=5, pressure=0, eta=1, gamma=1,
).detach().cpu().numpy()
assert samples.shape == (1, 3, 20, 20)
assert np.isfinite(samples).all()
np.save(args.output, samples)
occupancy = (samples[:, 0] >= 0).astype(np.uint8)
print('Output shape:', samples.shape)
print('Raw coordinate range:', samples[:, 0].min(), samples[:, 0].max())
print('Occupied sites:', occupancy.sum())
print('Density:', occupancy.mean())
print('Code controls are not a verified physical parameter mapping.')
