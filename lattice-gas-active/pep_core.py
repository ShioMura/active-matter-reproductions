from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import copy
import hashlib
import json
import platform
import time
import uuid
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PROJECT_DIR = Path.cwd()
OUTPUT_ROOT = PROJECT_DIR / "runs_pep"
DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
DIRECTION_NAMES = ["+row", "-row", "+column", "-column"]

print("Python:", platform.python_version(), "NumPy:", np.__version__)
print("Working directory:", PROJECT_DIR)


PARENT_NOTEBOOK_SHA256 = 'ec8e2fd1d3e26619db82a6033bead133ea8586953e0331f3c865b567a6a578b4'
REUSED_FUNCTION_SHA256 = {'count_boundary_bonds': 'fd471f3159b5d84b7cc7765fd621007d43c49d67e9fadf5003881c42dad5da14', 'lowest_mode_structure': 'd26be8bf002e972d92061d349791189a8300b6c9456297c254c228da87707028', 'periodic_cluster_observables': 'adc5b8dfd54c014c92f10c066786f5ef30f498216a96c837f7520617bc9e93de'}
PROVENANCE = {
    "parent_filename": "lattice_gas_mc(1).ipynb",
    "parent_notebook_sha256": PARENT_NOTEBOOK_SHA256,
    "reused_function_sha256": REUSED_FUNCTION_SHA256,
    "model_reference": "Soto and Golestanian, PRE 89, 012706 (2014), Section II",
    "doi": "10.1103/PhysRevE.89.012706",
    "reused": "Periodic geometry, fixed N, and three unmodified structural functions",
    "new": "Particle orientations, PEP updates, state/checkpoint handling, local-density measure",
    "removed_from_active_dynamics": "Attraction energy, Metropolis acceptance, chemical potential, PT",
    "previous_reference_status": "NOT_READY_PRECISION_SCREEN_NOT_MET; unchanged",
}


def make_state(L, density, rng, initial="random"):
    if isinstance(L, bool) or int(L) != L or L < 3:
        raise ValueError("L must be an integer >= 3.")
    L = int(L)
    if not np.isfinite(density) or not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1].")
    target = float(density) * L * L
    N = int(round(target))
    if not np.isclose(target, N, rtol=0, atol=1e-9):
        raise ValueError("Choose a density with integer density * L**2; no silent rounding.")
    if initial == "random":
        flat = rng.choice(L * L, size=N, replace=False)
    elif initial == "stripe":
        flat = np.arange(N)  # Filled rows; a partial final row is allowed.
    else:
        raise ValueError("initial must be 'random' or 'stripe'.")
    positions = np.column_stack(np.unravel_index(flat, (L, L))).astype(np.int64)
    owner = np.full((L, L), -1, dtype=np.int64)
    if N:
        owner[positions[:, 0], positions[:, 1]] = np.arange(N)
    state = dict(L=L, N=N, owner=owner, positions=positions,
                 orientation=rng.integers(0, 4, size=N, dtype=np.int8))
    assert_state(state)
    return state


def assert_state(state):
    L, N = state["L"], state["N"]
    pos, owner, ori = state["positions"], state["owner"], state["orientation"]
    assert owner.shape == (L, L) and pos.shape == (N, 2) and ori.shape == (N,)
    assert np.all((pos >= 0) & (pos < L))
    assert np.all((ori >= 0) & (ori < 4))
    assert np.all((owner >= -1) & (owner < N))
    assert np.count_nonzero(owner >= 0) == N
    if N:
        assert len(np.unique(pos[:, 0] * L + pos[:, 1])) == N
        assert np.array_equal(owner[pos[:, 0], pos[:, 1]], np.arange(N))
    return True


def occupancy(state):
    return (state["owner"] >= 0).astype(np.int8)


def orientation_field(state):
    field = np.full((state["L"], state["L"]), -1, dtype=np.int8)
    if state["N"]:
        p = state["positions"]
        field[p[:, 0], p[:, 1]] = state["orientation"]
    return field


def attempt_event(state, particle_id, tumble, new_direction):
    """One already-selected event. Tumble precedes the exclusion test."""
    if tumble:
        state["orientation"][particle_id] = new_direction
    i, j = state["positions"][particle_id]
    di, dj = DIRECTIONS[state["orientation"][particle_id]]
    k, ell = (i + di) % state["L"], (j + dj) % state["L"]
    if state["owner"][k, ell] >= 0:
        return False
    state["owner"][i, j] = -1
    state["owner"][k, ell] = particle_id
    state["positions"][particle_id] = (k, ell)
    return True


def pep_sweep(state, alpha, rng):
    """N asynchronous attempts, particle selections with replacement."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    N = state["N"]
    if N == 0:
        return 0, 0
    # Pre-draw random events; APPLY them sequentially to the evolving state.
    ids = rng.integers(0, N, size=N)
    tumble_uniforms = rng.random(N)
    new_dirs = rng.integers(0, 4, size=N)
    moved = 0
    for pid, u, direction in zip(ids, tumble_uniforms, new_dirs):
        moved += attempt_event(state, int(pid), u < alpha, int(direction))
    return int(moved), N


def count_boundary_bonds(n):
    """Count occupied-empty nearest-neighbor bonds once each."""
    return int(
        np.count_nonzero(n != np.roll(n, -1, axis=0))
        + np.count_nonzero(n != np.roll(n, -1, axis=1))
    )

def lowest_mode_structure(n):
    """Mean S(k)=|FFT(n-rho)|^2/M over the four smallest axial modes."""
    fluctuation = n.astype(float) - n.mean()
    spectrum = np.abs(np.fft.fft2(fluctuation)) ** 2 / n.size
    return float(
        (
            spectrum[1, 0] + spectrum[-1, 0]
            + spectrum[0, 1] + spectrum[0, -1]
        ) / 4
    )

def periodic_cluster_observables(state):
    """Measure occupied clusters and closed-path winding on a 2D torus."""
    a = np.asarray(state)
    if (
        a.ndim != 2
        or min(a.shape) < 3
        or not np.isin(a, [0, 1]).all()
    ):
        raise ValueError(
            "A binary 2D array with each dimension >= 3 is required."
        )

    occupied = a == 1
    shape = np.array(a.shape, dtype=int)
    labels = np.full(a.shape, -1, dtype=int)
    lift = np.zeros((*a.shape, 2), dtype=int)
    components = []

    for root in zip(*np.nonzero(occupied)):
        if labels[root] >= 0:
            continue

        cid = len(components)
        labels[root] = cid
        stack = [root]
        size = 0
        wraps = np.zeros(2, dtype=bool)
        basis = None
        rank = 0

        while stack:
            i, j = stack.pop()
            size += 1

            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (
                    (i + di) % a.shape[0],
                    (j + dj) % a.shape[1],
                )
                if not occupied[neighbor]:
                    continue

                proposed = lift[i, j] + (di, dj)

                if labels[neighbor] < 0:
                    labels[neighbor] = cid
                    lift[neighbor] = proposed
                    stack.append(neighbor)
                else:
                    assert labels[neighbor] == cid

                    # Two paths reach the same periodic site.
                    mismatch = proposed - lift[neighbor]
                    assert np.all(mismatch % shape == 0)
                    winding = mismatch // shape

                    if np.any(winding):
                        wraps |= winding != 0

                        if basis is None:
                            basis = winding.copy()
                            rank = 1
                        elif (
                            basis[0] * winding[1]
                            - basis[1] * winding[0]
                        ) != 0:
                            rank = 2

        components.append({
            "size": size,
            "wrap_axis0": bool(wraps[0]),
            "wrap_axis1": bool(wraps[1]),
            "winding_rank": rank,
        })

    n = int(occupied.sum())
    largest = max((c["size"] for c in components), default=0)
    assert sum(c["size"] for c in components) == n

    return {
        "N": n,
        "n_clusters": len(components),
        "f_max": largest / n if n else 0.0,
        "components": components,
    }


def local_density_field(n, width=5):
    if int(width) != width or width < 1 or width > min(n.shape):
        raise ValueError("width must be an integer between 1 and L.")
    w = int(width)
    total = np.zeros(n.shape, dtype=float)
    for di in range(w):
        for dj in range(w):
            total += np.roll(n, (di, dj), axis=(0, 1))
    return total / w**2


def measure(state, window=5):
    n = occupancy(state)
    clusters = periodic_cluster_observables(n)
    local = local_density_field(n, min(window, state["L"]))
    S = lowest_mode_structure(n)
    N, M = state["N"], n.size
    # Exact nonzero-wavevector mean for uniform fixed-N occupations (SEP).
    sep_S = N * (M - N) / (M * (M - 1))
    return dict(density=float(n.mean()), S_low=S, sep_S_low_reference=sep_S,
                f_max=clusters["f_max"], n_clusters=clusters["n_clusters"],
                boundary=count_boundary_bonds(n),
                local_density_variance=float(local.var()),
                wraps_any=any(c["winding_rank"] > 0 for c in clusters["components"]))


def run_implementation_checks():
    checks = {}
    rng = np.random.default_rng(2026111130)
    s = make_state(5, 1/25, rng)
    s["owner"][:] = -1
    s["positions"][0] = (0, 4)
    s["owner"][0, 4] = 0
    s["orientation"][0] = 2
    assert attempt_event(s, 0, False, 1)
    assert tuple(s["positions"][0]) == (0, 0)
    assert_state(s)
    assert attempt_event(s, 0, True, 1)
    assert tuple(s["positions"][0]) == (4, 0)
    checks["periodic_move_and_tumble_before_hop"] = True

    full = make_state(5, 1.0, rng)
    previous = full["positions"].copy()
    new_direction = (int(full["orientation"][0]) + 1) % 4
    assert not attempt_event(full, 0, True, new_direction)
    assert full["orientation"][0] == new_direction
    assert np.array_equal(full["positions"], previous)
    assert pep_sweep(full, 1.0, rng)[0] == 0
    assert_state(full)
    checks["exclusion_and_blocked_tumbling"] = True

    s = make_state(7, 1/49, rng)
    initial_pos = s["positions"].copy()
    initial_ori = s["orientation"].copy()
    for _ in range(11):
        moved, attempts = pep_sweep(s, 0, rng)
        assert moved == attempts == 1
    assert np.array_equal(s["orientation"], initial_ori)
    assert np.array_equal(s["positions"], (initial_pos + 11*DIRECTIONS[initial_ori]) % 7)
    empty = make_state(5, 0.0, rng)
    assert pep_sweep(empty, 0.1, rng) == (0, 0)
    checks["ballistic_and_empty_limits"] = True

    s = make_state(7, 20/49, rng)
    ref_positions = [tuple(p) for p in s["positions"]]
    ref_directions = s["orientation"].copy()
    reference_vectors = ((1,0),(-1,0),(0,1),(0,-1))
    for _ in range(2000):
        pid = int(rng.integers(s["N"]))
        tumble, newdir = bool(rng.random() < 0.2), int(rng.integers(4))
        if tumble:
            ref_directions[pid] = newdir
        r, c = ref_positions[pid]
        dr, dc = reference_vectors[int(ref_directions[pid])]
        target = ((r+dr) % 7, (c+dc) % 7)
        expected = target not in set(ref_positions)
        if expected:
            ref_positions[pid] = target
        assert attempt_event(s, pid, tumble, newdir) == expected
        assert np.array_equal(s["positions"], ref_positions)
        assert np.array_equal(s["orientation"], ref_directions)
        assert_state(s)
    checks["independent_reference_2000_events"] = True

    states = list(combinations(range(9), 2))
    index = {x:i for i,x in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    mean_structure = []
    for i, sites in enumerate(states):
        grid = np.zeros((3,3), dtype=np.int8)
        grid.flat[list(sites)] = 1
        mean_structure.append(lowest_mode_structure(grid))
        for start in sites:
            r,c = divmod(start,3)
            for dr,dc in reference_vectors:
                target = ((r+dr)%3)*3 + (c+dc)%3
                result = sites if target in sites else tuple(sorted((set(sites)-{start})|{target}))
                P[i,index[result]] += 1/8
    pi = np.full(len(states), 1/len(states))
    assert np.allclose(P.sum(axis=1),1,rtol=0,atol=1e-14)
    assert np.allclose(P,P.T,rtol=0,atol=1e-14)
    assert np.allclose(pi@P,pi,rtol=0,atol=1e-14)
    assert np.isclose(np.mean(mean_structure),2*7/(9*8),rtol=0,atol=1e-14)
    checks["exact_SEP_36_states_and_structure_reference"] = True

    for alpha in (0.0,0.05,1.0):
        Q = (1-alpha)*np.eye(4)+alpha*np.ones((4,4))/4
        assert np.allclose(Q@DIRECTIONS,(1-alpha)*DIRECTIONS)
    checks["analytic_orientation_kernel"] = True

    seam = np.zeros((8,8),dtype=np.int8);seam[3,[0,7]]=1
    c = periodic_cluster_observables(seam)
    assert c["f_max"]==1 and c["n_clusters"]==1
    assert c["components"][0]["winding_rank"]==0
    stripe = np.zeros((8,8),dtype=np.int8);stripe[:,2:6]=1
    c = periodic_cluster_observables(stripe)
    assert c["components"][0]["wrap_axis0"] and not c["components"][0]["wrap_axis1"]
    for a in (seam,stripe):
        assert np.isclose(local_density_field(a,3).mean(),a.mean())
        assert np.isclose(lowest_mode_structure(a),lowest_mode_structure(np.rot90(a)))
    checks["periodic_structure_known_answers"] = True

    s = make_state(8,0.5,rng)
    restored = copy.deepcopy(s)
    replay_rng = np.random.default_rng()
    replay_rng.bit_generator.state = json.loads(json.dumps(rng.bit_generator.state))
    for _ in range(20):
        pep_sweep(s,0.05,rng)
        pep_sweep(restored,0.05,replay_rng)
    for k in ("positions","orientation","owner"):
        assert np.array_equal(s[k],restored[k])
    checks["state_and_rng_replay"] = True
    checks["status"] = "IMPLEMENTATION_CHECKS_PASSED"
    return checks

CHECK_RESULTS = run_implementation_checks()
print(json.dumps(CHECK_RESULTS, indent=2))
