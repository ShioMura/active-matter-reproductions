# Active-matter reproductions

Research notebooks, recorded experiments, and reproducibility diagnostics for **Thermomaps Ising** and **ExpTM lattice gas**.

**Status:** this is an ongoing reproduction study. The repository records successful execution and repeatability checks alongside unresolved quantitative discrepancies. It does **not** establish a complete reproduction of the published results or a validated phase diagram.

## Start here

| Project | Recommended starting point | What it contains |
| --- | --- | --- |
| Ising | [Seed sensitivity and corrected MC reference](ising/ising_seed_check.ipynb) | Sampling repeatability, controlled training-seed comparisons, low-temperature trapping diagnostics, Wolff validation, and updated MC–TM comparisons |
| Ising | [Main experiment notebook](ising/thermomaps_ising.ipynb) | Simulation, training, generation, and historical comparisons |
| ExpTM | [Reviewed reproduction record](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/GC_ising_reproduction_reviewed.ipynb) | Recorded density results, checkpoint comparisons, three training seeds, and source/pipeline audit |
| ExpTM | [One-sample replay](exptm/rerun_one_sample.py) | Script entry point using hash-verified public data and checkpoint assets |

Open the notebooks to read their saved figures and summaries before running cells. Embedded outputs are records of particular executions; they are not evidence that every cell has been rerun in the current checkout.

## Ising: current findings

The study uses an 8×8 ferromagnetic Ising lattice with periodic boundaries.

- Reusing a sampling seed reproduced the fixed-checkpoint samples exactly in the recorded CPU environment.
- Two controlled training seeds produced substantially different predictions. Sampling repeatability therefore does not establish training robustness.
- The original single-spin MC reference became trapped in a stripe at low temperature. A separately implemented Wolff sampler was checked against exact 3×3 enumeration and used to rebuild the reference.
- The corrected comparison removes the observed low-temperature reference anomaly, but the tested TM checkpoint remains more ordered than the MC reference at higher temperatures.

These checks apply to the recorded configurations and protocols. Equilibration, independent validation data, and robustness across training runs still require attention. See the [seed-study notebook](ising/ising_seed_check.ipynb) for numerical results and uncertainty definitions.

## ExpTM: current findings

The recorded experiments compare 10,000-sample training, a fixed 2,000-sample subset with training seeds 42–44, and a public checkpoint.

Under the common recorded sampling protocol—T control 5, p controls −1/0/+1, 64 samples per control, batch size 16, and coordinate decoding at zero—the public checkpoint gives:

| p control | Generated mean density | Full input-data mean density |
| ---: | ---: | ---: |
| −1 | 0.3031250 | 0.0481045 |
| 0 | 0.4993359 | No reference supplied |
| +1 | 0.7245313 | 0.9515275 |

The endpoint mismatch remains unresolved. Changing T control from 5 to 1 did not remove it. The code controls have not been verified as the paper's physical temperature and chemical-potential mapping; the midpoint is not assigned an assumed target of 0.5.

Additional diagnostic notebooks:

| Notebook | Scope |
| --- | --- |
| [Training and checkpoint experiments](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/GC_ising_reproduction.ipynb) | Original working record; prefer the reviewed version for interpretation |
| [Reverse-process trace](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/GC_public_reverse_trace.ipynb) | Continuous coordinates along the reverse chain, with instrumented-versus-plain replay checks |
| [Paired denoising probe](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/GC_public_paired_denoising.ipynb) | Forward-noised versus clamped auxiliary channels with matched coordinate inputs and RNG states |
| [Selected-time profile](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/GC_public_time_profile.ipynb) | Denoising and reconstruction errors at selected diffusion times |
| [Exact 2×2 reference](exptm/author_source/GC_Lattice_gas/expTM_data_genenration/Exact_2x2_Reverse_Reference.ipynb) | Self-contained NumPy enumeration of a 16-state conditional reverse mixture |

The probes document large reconstruction errors at the highest tested noise level and sensitivity to auxiliary-channel treatment. They do not isolate a unique cause of the full sampling discrepancy. The 2×2 calculation is an independent mathematical reference under its stated coupling and illustrative clean distribution—not a reproduction of the 20×20 data or a validated replacement sampler.

## Running and environment requirements

Use **separate virtual environments and notebook kernels** for `ising/` and `exptm/`: both contain different packages named `tm`. Restart the kernel when switching projects.

The packaged dependency sets are [Ising requirements](ising/requirements.txt) and [ExpTM requirements](exptm/requirements.txt). Earlier portable setup used Python 3.11; the newer ExpTM diagnostic records use **Python 3.12.11 and PyTorch 2.5.1+cpu**. Some diagnostics explicitly assert that CPU version and source hashes. These environments should not be treated as interchangeable for exact replay.

For the script-based ExpTM replay, run inside `exptm/` with its environment activated:

```bash
python -m pip install -r requirements.txt
python fetch_assets.py
python fetch_assets.py --check-only
python rerun_one_sample.py
```

The asset fetcher downloads the three pinned inputs listed in [assets.json](exptm/assets.json) and verifies their sizes and SHA-256 hashes. The replay writes to a new `runs/` directory. Notebook execution additionally needs Jupyter and an appropriate kernel.

Before rerunning notebooks:

- Update project roots, checkpoint locations, saved-run paths, and interpreter checks for your machine. Several uploaded notebooks retain absolute Windows paths.
- In `ising_seed_check.ipynb`, the four `RUN_*` switches are currently **True**, despite the presentation-mode prose. Set all four to `False` if you only want to view the study without rerunning experiments.
- In the reviewed ExpTM notebook, training and resampling switches default to `False`, but some cells still read external saved files. Disabled training does not make all cells self-contained.
- The main Ising notebook retains upstream clone/install cells. Inspect these before execution; use the intended local source and kernel.
- Local trained checkpoints and the newer raw experiment directories are not fully included. Embedded plots do not replace the data and model files needed for replay.

The exact 2×2 reference is self-contained and requires NumPy, with no checkpoint or project paths.

## Records and provenance

Historical reports, source snapshots, and selected outputs are retained. Older subfolder READMEs and validation manifests describe earlier preparation stages and may not cover the newly uploaded studies. Treat their scope and dates explicitly; they do not certify the whole current repository.

See [PROVENANCE.json](PROVENANCE.json), [VALIDATION.json](VALIDATION.json), and [ExpTM source pin and license record](exptm/SOURCE_PIN_AND_LICENSE.json). Third-party code retains its original provenance and applicable licensing terms; this collection does not assign a new license to that work.
