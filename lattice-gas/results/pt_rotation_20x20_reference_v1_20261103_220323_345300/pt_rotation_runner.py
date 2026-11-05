"""Fixed-budget PT + independent rotations. Importing never starts simulation.

The supplied update/cluster implementation is preserved in the adjacent module.
Each committed NPZ is self-contained, atomic, and linked to its predecessor.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
DEP_PATH = HERE / "pt_rotation_dependencies.py"
_spec = importlib.util.spec_from_file_location("_pt_rotation_frozen_dependencies", DEP_PATH)
dep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dep)
CATEGORIES = ("none", "axis0_only", "axis1_only", "both_projections_rank1", "rank2_present")
LOADED_SOURCE_SHA256 = {
    "pt_rotation_runner.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "pt_rotation_dependencies.py": hashlib.sha256(DEP_PATH.read_bytes()).hexdigest(),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def default_config():
    return {
        "protocol": "PT_ROTATION_20X20_REFERENCE_V1",
        "L": 20, "N": 200, "epsilon": 1.0, "mu": 0.0,
        "boundary": "periodic",
        "temperatures": [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0],
        "seeds": {"random": [1742, 1743, 1744], "stripe": [1745, 1746, 1747],
                  "droplet": [1748, 1749, 1750]},
        "cycles": 50000, "initial_exclusion": 10000,
        "analysis_windows": [[10001, 30000], [30001, 50000]],
        "checkpoint_every": 5000, "snapshot_every": 10, "check_every": 100,
        "scalar_every": 1, "progress_every": 1000,
        "rng": {"bit_generator": "PCG64", "initialization": 0, "updates": 1, "rotations": 2},
        "rotation": "After each PT cycle, independent uniform k in {0,1,2,3} in EVERY slot; copy array",
        "sweep": "L**2 uniform ordered distinct-site pair proposals including no-change moves, per slot",
        "observables": ["U", "S_low", "D", "abs_D", "q"],
        "shape_categories": list(CATEGORIES),
        "screens": {
            "rank_rhat_max": 1.01, "pooled_bulk_ess_min": 1000,
            "block_cycles": [500, 1000, 2000, 4000],
            "per_chain_max_block_mcse": {"U": 0.25, "S_low": 0.125, "abs_D": 0.5, "f_max": 0.001},
            "joint": {"variables": ["U", "S_low"], "U_edges": [-400.5 + 2*i for i in range(202)],
                      "S_low_edges": list(range(102)), "within_chain_window_tv_max": 0.10,
                      "between_chain_full_window_tv_max": 0.10,
                      "interpretation": "Empirical histogram consistency, not distance to true equilibrium"},
            "shape_probability_mcse_target_per_chain": 0.001,
            "shape_probability_scope": "Exploratory: report per-chain block MCSE and occupied-block counts. Constant/rare traces remain UNRESOLVED; no shape-probability PASS based on MCSE alone.",
            "shape_probability_categories": ["none", "single_axis", "both_projections_rank1", "rank2_present"],
            "q_scope": "Rotation symmetry/execution diagnostic only; excluded from mixing qualification",
        },
        "stopping": "Stop at 50000 cycles per run regardless of diagnostics; no automatic extension",
        "scope": "Specified-observable reference candidate. Previous NOT READY decisions unchanged; no full-configuration or physical-dynamics qualification.",
    }


def budget(config):
    runs = sum(map(len, config["seeds"].values()))
    slots = len(config["temperatures"])
    return {"runs": runs, "cycles_per_run": config["cycles"],
            "within_slot_proposals": runs * config["cycles"] * slots * config["L"]**2,
            "swap_attempts": runs * config["cycles"] * (slots - 1),
            "rotation_calls_including_identity": runs * config["cycles"] * slots,
            "segments": runs * (config["cycles"] // config["checkpoint_every"]),
            "post_exclusion_snapshots_per_run": (config["cycles"] - config["initial_exclusion"]) // config["snapshot_every"]}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


def code_hashes():
    current = {"pt_rotation_runner.py": digest(__file__), "pt_rotation_dependencies.py": digest(DEP_PATH)}
    require(current == LOADED_SOURCE_SHA256, "Source changed after import; restart the kernel and load the frozen modules")
    return current


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(value, indent=2, allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path, arrays):
    path = Path(path)
    require(not path.exists(), f"Refusing to overwrite committed segment: {path}")
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def direct_energy(state, epsilon=1.0):
    a = np.asarray(state, dtype=np.int64)
    return float(-epsilon * np.sum(a * (np.roll(a, -1, 0) + np.roll(a, -1, 1))))


def observables(state, energy):
    a = np.asarray(state, dtype=float)
    spectrum = np.abs(np.fft.fft2(a - a.mean()))**2 / a.size
    s0 = (spectrum[1, 0] + spectrum[-1, 0]) / 2
    s1 = (spectrum[0, 1] + spectrum[0, -1]) / 2
    d = float(s0 - s1)
    return [float(energy), float((s0 + s1)/2), d, abs(d), float(d > 0) + 0.5*float(d == 0)]


def make_state(kind, config, rng):
    L, N = config["L"], config["N"]
    a = np.zeros((L, L), dtype=np.int8)
    if kind == "random":
        a.flat[rng.choice(L*L, N, replace=False)] = 1
    elif kind == "stripe":
        require(N % L == 0, "Stripe requires a whole number of columns")
        a[:, :N//L] = 1
    elif kind == "droplet":
        width = int(np.ceil(np.sqrt(N)))
        full_rows, remainder = divmod(N, width)
        height = full_rows + int(remainder > 0)
        require(width < L and height < L, "Compact droplet must fit without wrapping")
        r, c = (L-height)//2, (L-width)//2
        a[r:r+full_rows, c:c+width] = 1
        if remainder:
            a[r+full_rows, c:c+remainder] = 1
    else:
        raise ValueError(kind)
    require(int(a.sum()) == N, "Initialization particle count mismatch")
    return a


def check_state(states, energies, ids, temperatures, config):
    require(np.array_equal(temperatures, config["temperatures"]), "Temperature slots changed")
    require(np.array_equal(np.sort(ids), np.arange(len(states))), "Replica IDs are not a permutation")
    for i, state in enumerate(states):
        require(state.shape == (config["L"], config["L"]), "Wrong lattice shape")
        require(np.isin(state, [0, 1]).all() and int(state.sum()) == config["N"], "Occupation conservation failed")
        require(abs(direct_energy(state, config["epsilon"]) - energies[i]) <= 1e-10, "Cached energy mismatch")
        for other in states[:i]:
            require(not np.shares_memory(state, other), "Aliased configurations")


def _create(parent, config):
    """Internal configurable entry for bounded implementation tests only."""
    require(config["cycles"] % config["checkpoint_every"] == 0, "Incomplete segment budget")
    require(config["checkpoint_every"] % config["snapshot_every"] == 0, "Snapshot alignment")
    folder = Path(parent).resolve() / (config["protocol"].lower() + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config, "budget": budget(config), "environment": environment(), "source_sha256": code_hashes()}
    manifest["protocol_sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    for name in code_hashes():
        shutil.copyfile(HERE / name, folder / name)
    atomic_json(folder / "protocol.json", manifest)
    print("Created:", folder, flush=True)
    print(json.dumps(manifest["budget"], indent=2), flush=True)
    print("No simulation started. Keep this exact path for resume.", flush=True)
    return folder


def create_campaign(parent="results"):
    """Freeze the production protocol in a new directory, without running MC."""
    return _create(parent, default_config())


def read_manifest(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    expected = manifest["protocol_sha256"]
    contents = {k: v for k, v in manifest.items() if k != "protocol_sha256"}
    require(hashlib.sha256(canonical(contents).encode()).hexdigest() == expected, "Frozen protocol was modified")
    require(manifest["source_sha256"] == code_hashes(), "Running source differs from frozen source; use the campaign's saved modules")
    for name, sha in manifest["source_sha256"].items():
        require(digest(folder / name) == sha, f"Frozen source changed: {name}")
    for key in ("python", "numpy"):
        require(environment()[key] == manifest["environment"][key], f"Resume requires the original {key} version")
    return manifest


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _initialize(run_dir, kind, seed, manifest):
    config = manifest["config"]
    init = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0])))
    update = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    rotation = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2])))
    init_rng_start = canonical(init.bit_generator.state)
    states = [make_state(kind, config, init) for _ in config["temperatures"]]
    energies = np.array([direct_energy(a, config["epsilon"]) for a in states])
    ids = np.arange(len(states))
    check_state(states, energies, ids, np.array(config["temperatures"]), config)
    meta = {"protocol_sha256": manifest["protocol_sha256"], "kind": kind, "seed": seed,
            "start": 0, "end": 0, "previous_sha256": None, "elapsed_seconds": 0.0}
    arrays = {"meta_json": canonical(meta), "final_states": np.stack(states), "final_energies": energies,
              "final_ids": ids, "update_rng_end_json": canonical(update.bit_generator.state),
              "rotation_rng_end_json": canonical(rotation.bit_generator.state),
              "initialization_rng_start_json": init_rng_start,
              "initialization_rng_end_json": canonical(init.bit_generator.state)}
    atomic_npz(run_dir / "initial.npz", arrays)


def _recover(run_dir, kind, seed, manifest):
    """Verify all committed files in order, then return the last checkpoint."""
    config = manifest["config"]
    initial = run_dir / "initial.npz"
    if not initial.exists():
        require(not list(run_dir.glob("segment_*.npz")), "Segments without initial checkpoint")
        _initialize(run_dir, kind, seed, manifest)
    previous_path = initial
    previous = load_npz(initial)
    m0 = json.loads(str(previous["meta_json"]))
    require((m0["kind"], m0["seed"], m0["protocol_sha256"]) == (kind, seed, manifest["protocol_sha256"]), "Initial provenance mismatch")
    end = 0
    elapsed = 0.0
    for path in sorted(run_dir.glob("segment_*.npz")):
        segment = load_npz(path)
        meta = json.loads(str(segment["meta_json"]))
        next_end = end + config["checkpoint_every"]
        require(path.name == f"segment_{end:06d}_{next_end:06d}.npz", "Segment gap or unexpected filename")
        require(meta["start"] == end and meta["end"] == next_end <= config["cycles"], "Invalid cycle coverage")
        require((meta["kind"], meta["seed"], meta["protocol_sha256"]) == (kind, seed, manifest["protocol_sha256"]), "Segment provenance mismatch")
        require(meta["previous_sha256"] == digest(previous_path), "Predecessor file hash mismatch")
        for start_key, end_key in (("start_states", "final_states"), ("start_energies", "final_energies"), ("start_ids", "final_ids"),
                                   ("update_rng_start_json", "update_rng_end_json"), ("rotation_rng_start_json", "rotation_rng_end_json")):
            require(np.array_equal(segment[start_key], previous[end_key]), f"Checkpoint join mismatch: {start_key}")
        require(np.array_equal(segment["cycles"], np.arange(end+1, next_end+1)), "Scalar coverage mismatch")
        require(np.array_equal(segment["snapshot_cycles"], np.arange(end+config["snapshot_every"], next_end+1, config["snapshot_every"])), "Snapshot coverage mismatch")
        require(np.array_equal(segment["energy_by_slot"][-1], segment["final_energies"]), "Final energy record mismatch")
        require(np.array_equal(segment["replica_ids_by_slot"][-1], segment["final_ids"]), "Final ID record mismatch")
        previous_path, previous, end = path, segment, next_end
        elapsed += meta["elapsed_seconds"]
    states = [a.copy() for a in previous["final_states"]]
    energies = previous["final_energies"].copy()
    ids = previous["final_ids"].copy()
    check_state(states, energies, ids, np.array(config["temperatures"]), config)
    update = np.random.Generator(np.random.PCG64())
    rotation = np.random.Generator(np.random.PCG64())
    update.bit_generator.state = json.loads(str(previous["update_rng_end_json"]))
    rotation.bit_generator.state = json.loads(str(previous["rotation_rng_end_json"]))
    return end, states, energies, ids, update, rotation, previous_path, elapsed


def _segment(folder, kind, seed, manifest, recovered):
    config = manifest["config"]
    start, states, energies, ids, update, rotation, previous_path, _ = recovered
    stop = start + config["checkpoint_every"]
    n = stop-start
    slots = len(states)
    temperatures = np.array(config["temperatures"], dtype=float)
    arrays = {"start_states": np.stack(states), "start_energies": energies.copy(), "start_ids": ids.copy(),
              "update_rng_start_json": canonical(update.bit_generator.state), "rotation_rng_start_json": canonical(rotation.bit_generator.state),
              "cycles": np.arange(start+1, stop+1), "cold_observables": np.empty((n, 5)),
              "energy_by_slot": np.empty((n, slots)), "replica_ids_by_slot": np.empty((n, slots), dtype=np.int16),
              "exchange_accepted": np.empty((n, slots-1), dtype=bool), "rotation_k": np.empty((n, slots), dtype=np.int8)}
    snap_cycles, snaps, fmax, nclusters, classes = [], [], [], [], []
    t0 = time.perf_counter()
    for j, cycle in enumerate(arrays["cycles"]):
        arrays["exchange_accepted"][j] = dep.replica_exchange_cycle(states, energies, ids, temperatures, update, epsilon=config["epsilon"])
        ks = rotation.integers(0, 4, size=slots)
        for slot, k in enumerate(ks):
            states[slot] = np.rot90(states[slot], int(k)).copy()
        arrays["rotation_k"][j] = ks
        if cycle % config["check_every"] == 0 or cycle == stop:
            check_state(states, energies, ids, temperatures, config)
        arrays["cold_observables"][j] = observables(states[0], energies[0])
        arrays["energy_by_slot"][j] = energies
        arrays["replica_ids_by_slot"][j] = ids
        if cycle % config["snapshot_every"] == 0:
            shape = dep.periodic_cluster_observables(states[0])
            snap_cycles.append(cycle)
            snaps.append(states[0].copy())
            fmax.append(shape["f_max"])
            nclusters.append(shape["n_clusters"])
            classes.append(CATEGORIES.index(dep.occupied_wrapping_class(shape["components"])))
        if cycle % config["progress_every"] == 0:
            print(f"{kind:7s} seed={seed} cycle={cycle}/{config['cycles']} cold U={energies[0]:.1f}", flush=True)
    meta = {"protocol_sha256": manifest["protocol_sha256"], "kind": kind, "seed": seed, "start": start, "end": stop,
            "previous_sha256": digest(previous_path), "elapsed_seconds": time.perf_counter()-t0,
            "within_slot_proposals": n*slots*config["L"]**2, "swap_attempts": n*(slots-1), "rotation_calls": n*slots}
    require(np.isfinite(arrays["cold_observables"]).all(), "Non-finite observable")
    for column, edges in ((0, config["screens"]["joint"]["U_edges"]), (1, config["screens"]["joint"]["S_low_edges"])):
        values = arrays["cold_observables"][:, column]
        require(np.all((values >= edges[0]) & (values < edges[-1])), "Observable outside frozen histogram range")
    arrays.update({"meta_json": canonical(meta), "final_states": np.stack(states), "final_energies": energies.copy(), "final_ids": ids.copy(),
                   "update_rng_end_json": canonical(update.bit_generator.state), "rotation_rng_end_json": canonical(rotation.bit_generator.state),
                   "snapshot_cycles": np.array(snap_cycles), "cold_snapshots": np.stack(snaps), "f_max": np.array(fmax),
                   "n_clusters": np.array(nclusters), "shape_class": np.array(classes, dtype=np.int8)})
    path = folder / f"segment_{start:06d}_{stop:06d}.npz"
    atomic_npz(path, arrays)
    print("Committed:", path.name, flush=True)


def run_campaign(folder, *, max_new_segments=None):
    """Run/resume only the frozen budget. Optional segment cap pauses safely.

    KeyboardInterrupt loses only the uncommitted segment, which is replayed
    deterministically on resume. Never run this concurrently on one directory.
    """
    folder = Path(folder).resolve()
    manifest = read_manifest(folder)
    if max_new_segments is not None:
        require(isinstance(max_new_segments, int) and max_new_segments >= 0, "Invalid segment cap")
    lock = folder / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Lock exists: {lock}. If the prior process crashed, first verify it has stopped, then remove ONLY this lock file.") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\ncreated={datetime.now(timezone.utc).isoformat()}\n")
    new_count = 0
    rows = []
    try:
        for kind, seeds in manifest["config"]["seeds"].items():
            for seed in seeds:
                run_dir = folder / f"{kind}_{seed}"
                run_dir.mkdir(exist_ok=True)
                recovered = _recover(run_dir, kind, seed, manifest)
                while recovered[0] < manifest["config"]["cycles"]:
                    if max_new_segments is not None and new_count >= max_new_segments:
                        print("Paused at a committed checkpoint. Resume with run_campaign(the_same_path).", flush=True)
                        return {"status": "PAUSED", "new_segments": new_count}
                    _segment(run_dir, kind, seed, manifest, recovered)
                    new_count += 1
                    recovered = _recover(run_dir, kind, seed, manifest)
                rows.append({"initial": kind, "seed": seed, "cycles": recovered[0], "committed_segment_seconds": recovered[-1],
                             "last_checkpoint_sha256": digest(recovered[-2])})
        report = {"status": "FIXED_BUDGET_EXECUTION_COMPLETE_ASSESSMENT_PENDING", "protocol_sha256": manifest["protocol_sha256"],
                  "budget": manifest["budget"], "runs": rows,
                  "scope": "Recorded bookkeeping checks passed. Statistical assessment pending; no automatic continuation or equilibrium qualification.",
                  "timing_scope": "Committed segment compute time; excludes discarded interrupted work, most IO, and validation overhead."}
        atomic_json(folder / "execution_summary.json", report)
        print(report["status"], flush=True)
        print("Saved:", folder / "execution_summary.json", flush=True)
        return report
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    print("Import this module from the supplied notebook. No simulation started.")
