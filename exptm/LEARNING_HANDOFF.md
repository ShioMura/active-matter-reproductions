# ExpTM learning handoff — one pinned lattice-gas checkpoint execution

This artifact transfers evidence; no communication with the separate learning chat is claimed. Read the final REPRODUCTION_REPORT.md and RUN_MANIFEST.json for actual outcomes. The executed local entry is `run_checkpoint.py`, a path/CPU/checkpoint adapter around the authors' `GC_ising_code.ipynb` cells1–7 and9–10. Training cell8 was not executed.

## Follow this short call sequence

| Step | Author file/function | Important shape / operation |
|---|---|---|
|1|GC_ising_code.ipynb cells1–4|Two supplied arrays `(5000,2,20,20)`; concatenate into10000 observations. Keep spin channel0; append auxiliaries. Final `(10000,3,20,20)` with literal temperature5 and normalized mean-control−1/+1.|
|2|tm/core/loader.py:Loader|Float32 data; original Gaussian dequantization0.01 and NormalTransform constructed. In this call graph, training fetches raw dequantized data and sampling decoder does not apply unstandardize.|
|3|tm/core/prior.py:GlobalEquilibriumHarmonicPrior.sample|Shape `(batch,3,20,20)`. Coordinate Gaussian mean=mean-control, variance=temperature. Auxiliary prior channels have mean0/variance1.|
|4|tm/architectures/unet_2d_mid_attn.py:Unet2D|dim32, multipliers1/2/4,3channels, groups8;3,808,739 parameters. Strict load checkpoint1.|
|5|tm/core/backbone.py:ConvBackbone|Nearest-exact20→32 spatial interpolation, network forward,32→20 interpolation. Original BB self-conditioning recipe retained.|
|6|tm/core/diffusion_process.py:VPDiffusion|100 polynomial artificial diffusion times; forward_kernel, reverse_kernel, reverse_step. These times are not physical MC dynamics.|
|7|tm/core/diffusion_model.py:SteeredDiffusionSampler|sample_loop→sample_batch→denoise_step→reverse_step; steer auxiliary channels to literalT5 and mean-control0 with gamma1, eta1.|
|8|GC_ising_code.ipynb:round_lattice|Channel0<0→empty0, otherwise occupied1; density=sum occupancies/400. This registered decoder is preserved.|

The published AIP counterpart uses Methods Eqs18–21 and Supplementary Algorithm1; the arXiv numbering below is kept explicit. The paper's grand-canonical discussion and Gaussian mean/variance construction are in arXiv2503.15080v1 SectionsII.4.2–II.4.3, Eqs15–18; the20×20 example is SectionIII.1/Fig2. SI Algorithm1 describes the data→tilted prior→reverse mapping. The code implements a discrete noise-prediction version; do not identify its artificial time with physical dynamics or infer ABP activity response.

## Inspect and rerun a small same-example calculation

From the extracted release root, install the recorded CPU environment with SETUP_AND_COMMANDS.md. Then:

```bash
./environment/venv/bin/python learning_example.py --output my_one_sample.npy
```

This optional one-sample script uses the same checkpoint and same native controls as the registered smoke test. Choose a new output filename; it refuses overwriting existing evidence. It has not been executed as an additional experiment in this release. It does not retrain a model or establish paper reproduction. For an entirely compute-free inspection, load `outputs/smoke/generated_samples.npy` and read its `(2,3,20,20)` shape and decoded densities.

Registered commands:

```bash
./environment/venv/bin/python run_checkpoint.py smoke
./environment/venv/bin/python run_checkpoint.py quantitative
```

These refuse existing stage outputs and are reproducibility records, not a request to rerun completed stages. The latter is the single256-sample diagnostic; do not increase sample counts or vary temperatures to make a nicer result. `verify_release.py` performs read-only integrity/summary checks.

## What was obtained and what was not

Smoke execution loaded a real compatible author checkpoint, ran all100 reverse steps and produced finite output. The256-sample registered output and conditional uncertainty are reported separately; no independent training, full Fig2 heatmap reproduction, parameter-generalization test or speedup benchmark was performed.

Important provenance findings: checkpoint49 is incompatible with the lattice-gas3-channel network; checkpoint1 is structurally compatible but lacks figure-run configuration. Current entry points disagree about temperature and training duration; provided arrays contain5000 observations per anchor, while the arXiv SI describes10000 each. `GC_original.py`/`GC_original.ipynb` also require raw accumulation inputs absent from this snapshot. Do not repair these scientific differences by guessing from a plausible picture.

One central mean near1/2 is intrinsically a weak test: particle-hole symmetry gives1/2 at the intended symmetry center regardless of positive temperature, and many incorrect distributions share that mean. It does not validate morphology or the learned full probability law. The only justified next step toward published-result reproduction is recovering the author figure-run bindings, not starting ABP or a new TM mechanism.
