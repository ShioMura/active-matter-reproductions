# PT + rotation: bounded 20x20 campaign

This package prepares and executes one new reference-candidate campaign. It does
not replace, extend, or qualify any historical dataset. The supplied
pt_rotation_dependencies.py is byte-identical to the uploaded source.

## Use

1. Extract the whole folder beside lattice_gas_mc.ipynb (inside lattice-gas).
2. Open run_pt_rotation_20x20.ipynb. Select the existing Python kernel.
3. Run its small implementation-check cell (NumPy is the only dependency).
4. Run the create-campaign cell ONCE, retain the printed full result directory.
5. Run the execution cell. It runs the fixed budget and stops; it does not assess
   statistical success. Avoid Run All in the old long notebook.

The notebook is independent of the old notebook's live globals. Keep the two .py
files together. The notebook creates results under the parent lattice-gas folder.
If you put this folder elsewhere, choose the output parent explicitly.

## Resume

For a normal notebook interrupt, call run_campaign with the SAME campaign folder.
A new kernel must import the module and set campaign_dir to the saved absolute
path; do not call create_campaign again. Recovering an incomplete segment replays
it from its preceding committed checkpoint, including both RNG streams.

There is one lock per campaign. A normal interrupt removes it. A killed kernel or
power failure can leave RUNNING.lock: verify that no process is using this
campaign before removing ONLY that file. Never delete trajectory segments to make
an integrity error disappear. Uncommitted *.tmp files are ignored.

Source and protocol hashes are checked. Python and NumPy versions must match the
campaign's recorded versions. Frozen source copies are placed in each campaign;
use those copies if the original package has been changed. Do not edit frozen
configurations to extend budgets or adjust criteria after looking at results.

## Frozen design

- L=20, N=200, epsilon=1, mu=0, periodic boundaries.
- Temperatures: .4,.45,.5,.55,.6,.7,.8,1.
- Random seeds 1742-1744; stripe 1745-1747; compact droplet 1748-1750.
- Each temperature starts with an independent array. Random arrays are generated
  independently; ordered shapes are initially identical across slots.
- Droplet is a centered 15-column compact patch: 13 full rows plus 5 sites.
  It is connected and non-wrapping, with exactly 200 occupied sites.
- Each cycle: 400 uniform distinct-site proposals per slot; even/odd adjacent
  exchanges; independent uniform quarter-turn rotation in every slot, including
  identity. Initialization/update/rotation use PCG64 streams [seed,0/1/2].
- 50,000 cycles per run; initial 10,000 excluded from analysis by design, without
  claiming that this establishes equilibration.
- Two measurement windows: 10,001-30,000 and 30,001-50,000.
- Every-cycle cold U,S_low,D,abs_D,q, slot energies, replica IDs, swaps, rotations.
- Cold configurations, f_max, cluster count, wrapping class every 10 cycles.
- Direct all-slot energy, conservation, ID and alias checks every 100 cycles
  and at every segment endpoint.
- Atomic complete segments every 5,000 cycles, 90 segments total; 1.44 billion
  within-slot proposals, 3.15 million swaps, 3.6 million rotation calls.

All FFT normalization uses the actual lattice size. Rotation improves access to
symmetry-related orientations; it does not itself change rotation-invariant
shape classes. q is therefore an execution/symmetry diagnostic, not a mixing gate.
This artificial sampler's trajectories are not local Kawasaki physical dynamics.

## Pre-execution assessment specification

See protocol_preview.json, and the frozen protocol.json produced on creation.
For U,S_low,abs_D,f_max: rank R-hat <=1.01; pooled bulk ESS >=1000;
maximum per-chain batch MCSE across 500,1000,2000,4000-cycle blocks <=
.25,.125,.5,.001 respectively. At 10-cycle snapshot spacing, the f_max block
lengths are 50,100,200,400 observations. Pooled precision cannot substitute for
per-chain precision.

For empirical P(U,S_low), use fixed bins of width 2 and 1 with the explicit
edges in the protocol. The domains cover the full physical range here
(U in [-400,0], S_low <=100; both bounds deliberately conservative).
Reject out-of-range observations rather than discard or rebin them. Compare
all per-chain half-window histograms (max TV <=.10), and all between-chain
full-measurement histograms (max TV <=.10). These are specified empirical
consistency tolerances, not bounds on distance to the true equilibrium law.

Shape probabilities merge the two single-axis classes for the rotation-invariant
assessment; retain their separate labels descriptively. Report per-chain batch
MCSE against an absolute target .001, and counts of blocks containing events.
Rare/constant traces remain UNRESOLVED. No numerical event-count criterion has
been validated, so this campaign does not automatically award a shape-probability
PASS merely for meeting MCSE. Zero observations do not imply zero probability.

This package performs execution and bookkeeping, not the statistical assessment.
The frozen screens are saved before any production samples are generated. A
separate read-only assessment follows the completed run. Even passing these
screens would not qualify the full configuration distribution or bound common
bias. Previous NOT READY decisions remain unchanged.

## Files and records

protocol.json contains the frozen configuration, software versions, source hashes,
and exact budget. A run subdirectory contains initial.npz and ten segment files.
The initial file stores all states and RNG states. Each segment stores start/end
states, cached energies and IDs, update/rotation RNG states, its predecessor's
SHA256, cycle coverage and observations. Scalar records are start+1 through end
(no duplicate boundary samples). Snapshots are likewise unique by cycle.
shape_class is encoded in the order in config.shape_categories; cold_observables
columns are config.observables. All NPZ files load with allow_pickle=False.

execution_summary.json is created only after every run finishes. Its status is
FIXED_BUDGET_EXECUTION_COMPLETE_ASSESSMENT_PENDING. Rerunning a completed campaign
adds no MC steps. Reported segment timings exclude discarded interrupted work
and most IO/validation overhead, so they are not total cost benchmarks.

## Validation performed here

implementation_checks.json records the tested environment and source hashes.
Checks include exhaustive 3x3 fixed-N energy changes, 20x20 initial geometries and
FFT/rotation invariants, and exact 4x4 trajectory/RNG equality across continuous,
checkpoint-paused and mid-segment-interrupted execution. Locks, changed protocol,
and missing segments are checked. These are implementation tests; no production
20x20 trajectory or equilibrium assessment was run here. Run the same small
checks once in your Windows Python 3.14.2 environment before production.
