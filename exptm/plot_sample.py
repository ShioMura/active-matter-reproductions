"""Plot a saved one-sample checkpoint replay without changing the sample."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".png")
    if output.exists():
        parser.error("Choose a new output filename.")
    samples = np.load(args.input, allow_pickle=False)
    if samples.shape != (1, 3, 20, 20) or not np.isfinite(samples).all():
        parser.error("Expected finite data with shape (1, 3, 20, 20).")
    raw = samples[0, 0]
    occupancy = (raw >= 0).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    limit = max(float(np.abs(raw).max()), 1e-12)
    im = axes[0].imshow(raw, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    axes[0].set_title("Raw generated coordinate")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].imshow(occupancy, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Binary occupancy | density = {occupancy.mean():.4f}")
    for ax in axes:
        ax.set_xlabel("Lattice x"); ax.set_ylabel("Lattice y")
    if np.all(samples[:, 1] == 5) and np.all(samples[:, 2] == 0):
        fig.suptitle("One generated sample | code controls: T=5, P=0")
    else:
        fig.suptitle("One generated sample")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
