# ExpTM checkpoint replay

Portable entry points: `fetch_assets.py`, `rerun_one_sample.py`, `plot_sample.py`.

The original `run_checkpoint.py`, reports, source snapshot and historical outputs are retained. `records/user_run/` contains the supplied one-sample scripts and output. Earlier reports refer to their original execution environment, not this Mac preparation.

The new runner uses the same author checkpoint (`model_1.pt`), preparation seed and sample seed, T=5 and P=0 code controls, and threshold decode >=0. It verifies the checkpoint and both input arrays before loading. Data/checkpoint assets come from the pinned upstream commit in `assets.json`; failed checks never overwrite existing files. New outputs go under `runs/`.

This is checkpoint replay, not independent model training. The physical meaning of the notebook control mapping and published Fig. 2 configuration remain unverified. The incompatible `model_49.pt` and paper PDFs are omitted from this runnable collection; their historical metadata remain in the reports. CPU is used on macOS.
