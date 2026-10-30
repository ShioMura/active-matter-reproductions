"""Replay the supplied one-sample example; leave historical evidence untouched."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from fetch_assets import ROOT, ensure_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New .npy file; defaults to a unique runs/ folder")
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / "sample.npy"
    output = output.expanduser().resolve()
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        parser.error("Choose a new output filename; existing outputs are preserved.")
    ensure_assets(check_only=True)
    from run_checkpoint import setup

    sampler, loader, metadata = setup()
    print("Input shape:", tuple(loader.data.shape))
    torch.manual_seed(2026101304)
    samples = sampler.sample_loop(
        num_samples=1, batch_size=1,
        temperature=5, pressure=0, eta=1, gamma=1,
    ).detach().cpu().numpy()
    assert samples.shape == (1, 3, 20, 20) and np.isfinite(samples).all()
    assert np.all(samples[:, 1] == 5) and np.all(samples[:, 2] == 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.save(stream, samples, allow_pickle=False)
    occupancy = samples[:, 0] >= 0
    metadata.update(sample_seed=2026101304, temperature=5, pressure=0, eta=1, gamma=1,
                    shape=list(samples.shape), density=float(occupancy.mean()),
                    numpy=np.__version__, physical_parameter_mapping_verified=False)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
    print("Saved:", output)
    print("Occupied sites:", int(occupancy.sum()), "Density:", float(occupancy.mean()))
    print("Code controls are not a verified physical parameter mapping.")


if __name__ == "__main__":
    main()
