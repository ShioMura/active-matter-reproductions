# Exact CPU setup and commands

Pinned source commit: `28f1252eed7cbaaf67d901b337ffee1885552fbe` (main inspected2026-10-13). Source snapshots are immutable byte-verified copies under author_source. Local adapter/code and generated outputs are separate. No older thermomaps-ising substitution or HOOMD execution.

```bash
python3 -m venv --system-site-packages environment/venv
./environment/venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
./environment/venv/bin/python -m pip install einops==0.8.1 tqdm==4.67.1
```

The executed environment inherited installed NumPy/SciPy/Pandas/scikit-learn/PyYAML/matplotlib from the base environment; exact versions and pip freeze are supplied. For a fresh computer use execution_requirements.txt to install the used scientific versions, in addition to the separate CPU torch command. The authors' unversioned requirements.txt is retained. slurm-workflow and unused CLI extras are not required by this inference call graph and were not installed. The venv binaries are not archived; reconstruct them from this specification.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 meter.py smoke_checkpoint ./environment/venv/bin/python run_checkpoint.py smoke
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 meter.py quantitative_checkpoint ./environment/venv/bin/python run_checkpoint.py quantitative
```

Already-completed stages refuse execution. Original command records in RESOURCE_LEDGER.json retain the actual workspace-relative invocation. Cumulative active numerical/setup wall/CPU limits are1800/7200seconds; GPU was unavailable and used0seconds. Each generation batch is saved immediately. Local seeds are in FROZEN_TARGET.json; initial/final RNG states are archived. These seeds are not represented as author training seeds.

Read-only verification:

```bash
./environment/venv/bin/python verify_release.py
```

A whole-notebook execution is not equivalent to this checkpoint-based call: it would train and run extra controls/heatmaps. It is outside this completed one-target release.
