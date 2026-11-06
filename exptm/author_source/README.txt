ExpTM 11x11 control-grid code

1. Extract this ZIP, keep its files together.
2. Open Run_Figure2_11x11.ipynb with your working ExpTM Python kernel.
3. Set PROJECT_ROOT to the existing reproduction directory containing author_source.
4. Run All. 121 cells x 16 samples = 1,936 samples; CPU only; no training.
5. For an interrupted run, set RESUME_OUTPUT to the original output path and rerun.

The archived author source, model_1.pt, LowMuT7.npy and HighMuT7.npy are REQUIRED and are NOT included here. No files from your existing project are modified.
Checkpoint SHA-256: 72125a26b5d136b8eeda782ff01fbecb99963b3ad58b8fce3f4494185f5419fc
Recorded environment: Python 3.12, PyTorch 2.5.1+cpu, numpy 2.3.5, einops 0.8.1.
Prefer the already working ExpTM environment. requirements-run.txt lists direct dependencies for a separate environment; a Mac is not promised byte-identical outputs to this CPU/Linux run.

Command line (quote paths containing spaces):
python run_heatmap.py --root YOUR_PROJECT_ROOT --output NEW_OUTPUT_DIRECTORY
python plot_heatmap.py NEW_OUTPUT_DIRECTORY
Resume:
python run_heatmap.py --root YOUR_PROJECT_ROOT --output SAME_OUTPUT_DIRECTORY --resume

Each point resets sampling seed 2026103101; the grid cells share random numbers.
T controls 2..10, p controls -1..1; 11 points along each axis; 100 diffusion steps; eta=gamma=1.
All 121 cells are actually sampled. No heatmap interpolation. The fixed 90-minute cap counts saved successful-cell sampling time and is checked between cells. Interrupted unfinished-cell work and setup are outside this recorded time.
Resume validates exact config/source/data/environment/runner identity and saved sample contents. A changed machine or environment requires a new output directory.

This is a preliminary Figure 2(b)-style control-grid diagnostic. Physical T/chemical-potential mapping and Figure 2 checkpoint identity remain unverified. No matched MC error map or phase-boundary claim.
