import numpy as np
import matplotlib.pyplot as plt

samples = np.load("my_new_sample.npy")
raw = samples[0, 0]
occupancy = (raw >= 0).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

limit = np.abs(raw).max()
image = axes[0].imshow(
    raw, cmap="coolwarm", vmin=-limit, vmax=limit,
    interpolation="nearest",
)
axes[0].set_title("Raw generated coordinate")
fig.colorbar(image, ax=axes[0], shrink=0.8)

axes[1].imshow(
    occupancy, cmap="binary", vmin=0, vmax=1,
    interpolation="nearest",
)
axes[1].set_title(f"Binary occupancy | density = {occupancy.mean():.4f}")

for ax in axes:
    ax.set_xlabel("Lattice x")
    ax.set_ylabel("Lattice y")

fig.suptitle("One generated sample | code controls: T=5, P=0")
plt.tight_layout()
plt.savefig("my_new_sample.png", dpi=200, bbox_inches="tight")
plt.show()