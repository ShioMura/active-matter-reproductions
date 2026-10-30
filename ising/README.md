# Thermomaps Ising

Open `thermomaps_ising.ipynb` with this folder's environment. `smoke_test.py` checks the simulation and ML interfaces on a small workload.

`records/thermomaps_ising_executed.ipynb` is the supplied executed notebook, unchanged. `thermomaps-root/` is the supplied source, unchanged. The runnable notebook clears old outputs, uses local paths and unique run folders, and lowers the generation batch from 5,000 to 50 while retaining 5,000 samples per temperature and 20 training epochs. Batching and seeding changes can change individual generated samples. It is not expected to reproduce the old output bit for bit.

The recorded T=0.3 trajectory was stuck in a stripe (one distinct saved configuration, |m|=0.5, E=-96). Its flat observables do not validate equilibrium. This diagnosis is retained; the sampling algorithm has not been silently changed. Random frame train/test splits also do not establish independent-history validation.
