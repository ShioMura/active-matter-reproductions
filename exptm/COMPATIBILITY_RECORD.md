# Compatibility and scientific-effect record

No author source file was edited. Author text files are checked against Git blob SHA-1 and SHA-256; downloaded data/checkpoints also match Git blobs and declared byte sizes.

| Item | Local action | Effect / limitation |
|---|---|---|
| README installation path | Use actual `GC_Lattice_gas/expTM_data_genenration` subtree on PYTHONPATH | Root `thermomaps-root/requirements.txt` does not exist. No scientific equation changes. |
| Missing raw accumulation files | Use already provided HighMuT7/LowMuT7 arrays through `GC_ising_code.ipynb` preprocessing | No new MC simulation or downsampling. Current supplied inventory5000+5000 differs from paper SI20000. |
| Notebooks / script disagree | Execute literal GC_ising_code cells1–7 and9–10, replacing training cell8 with checkpoint loading | NativeT5 recipe, not an inferred repair to T7. Published figure binding unresolved. |
| Highest numbered checkpoint49 | Strict unchanged3-channel network load fails; retain failure and 4-channel shape inspection | Did not resize weights, fabricate a4-channel lattice representation, train or run CO2. |
| Compatible checkpoint1 | Choose structurally compatible latest member of first-two larger-file family, before predictions | Strict dim32/channels3 matching. This is not model selection by density error. |
| CPU-only device | Pass `device='cpu'` to original `ConvBackbone.load_model` | Default device0 is unavailable. Weights and sampler unchanged. |
| Versioned environment | Python3.12, torch2.5.1+cpu, einops0.8.1, tqdm4.67.1; inherited scientific libraries recorded | Authors publish unversioned requirements; original numerical environment not exactly recovered. |
| Scheduler / training | No retraining; no scheduler or optimizer update | Independent training and speedup not tested. |
| Seed availability | Freeze local preparation/smoke/generation seeds; preserve RNG states | Author seed history not supplied; no bitwise published-output reproduction claim. |
| Loader normalization | Build and archive original NormalTransform, dequantization0.01 | Author training/sampling path does not call Loader.standardize/unstandardize on output spins; no new decoder/scaler. |
| Self-conditioning | Preserve BB `eval_mode='train', self_condition=True`, model default `self_condition=False` | No inference optimization or architecture change; stochastic draw consumption stays original recipe. |
| Package extras | Do not install slurm-workflow or unused CLI/Ising modules | Core selected inference call graph does not import these. No cluster submissions. |
| Resource ledger | Restore fetch-input receipt lost during overlapping setup final-write; subsequently serialize metered stages | All actual measured costs retained; conservative sum of stage wall may double-count overlap. Scientific outputs unaffected. |

No compatibility patch to the scientific author algorithms is needed; the local runner is a transparent path/checkpoint/device adapter. A missing figure configuration is an unresolved provenance dependency, not a compatibility fix to be invented.
