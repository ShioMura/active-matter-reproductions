import numpy as np


# ===== replica_exchange_cycle =====
def replica_exchange_cycle(
    states, energies, replica_ids, temperatures, rng, epsilon=1.0
):
    """One within-slot sweep followed by even/odd adjacent exchanges."""
    n_slots = len(temperatures)

    # Update each configuration at its current slot temperature.
    for slot, temperature in enumerate(temperatures):
        _, delta_u = nonlocal_sweep(
            states[slot],
            temperature=float(temperature),
            rng=rng,
            epsilon=epsilon,
        )
        energies[slot] += delta_u

    # Each adjacent edge is attempted once per cycle.
    accepted_edges = np.zeros(n_slots - 1, dtype=bool)

    for parity in (0, 1):
        for a in range(parity, n_slots - 1, 2):
            accepted_edges[a] = attempt_replica_exchange(
                states, energies, replica_ids, temperatures,
                a, a + 1, rng,
            )

    return accepted_edges


# ===== nonlocal_sweep =====
def nonlocal_sweep(n, temperature, rng, epsilon=1.0):
    """Attempt n.size uniformly proposed nonlocal exchanges."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")

    rows, cols = n.shape
    if min(rows, cols) < 3:
        raise ValueError("Both lattice dimensions must be >= 3.")

    num_sites = n.size
    moved = 0
    total_delta = 0.0

    for _ in range(num_sites):
        # Uniform ordered pair of distinct sites.
        site_a = int(rng.integers(num_sites))
        site_b = int(rng.integers(num_sites - 1))
        if site_b >= site_a:
            site_b += 1

        i, j = divmod(site_a, cols)
        k, l = divmod(site_b, cols)

        # Equal occupations produce a no-change attempt.
        if n[i, j] == n[k, l]:
            continue

        delta_u = nonlocal_swap_energy_change(
            n, i, j, k, l, epsilon
        )

        if delta_u <= 0 or rng.random() < np.exp(-delta_u / temperature):
            a, b = int(n[i, j]), int(n[k, l])
            n[i, j], n[k, l] = b, a
            moved += 1
            total_delta += delta_u

    # Changed configurations / all attempts, not just unlike pairs.
    return moved / num_sites, total_delta


# ===== nonlocal_swap_energy_change =====
def nonlocal_swap_energy_change(n, i, j, k, l, epsilon=1.0):
    """Interaction-energy change for exchanging two arbitrary sites."""
    rows, cols = n.shape
    if min(rows, cols) < 3:
        raise ValueError("Both lattice dimensions must be >= 3.")

    if (i, j) == (k, l):
        return 0.0

    a = int(n[i, j])
    b = int(n[k, l])
    if a == b:
        return 0.0

    def neighbor_sum(r, c):
        return (
            int(n[(r - 1) % rows, c])
            + int(n[(r + 1) % rows, c])
            + int(n[r, (c - 1) % cols])
            + int(n[r, (c + 1) % cols])
        )

    adjacent = (
        (j == l and ((i - k) % rows in (1, rows - 1)))
        or
        (i == k and ((j - l) % cols in (1, cols - 1)))
    )

    # A shared bond has unchanged product a*b after the exchange.
    # Exclude it from both neighbor sums when the sites are adjacent.
    neighbors_a = neighbor_sum(i, j) - int(adjacent) * b
    neighbors_b = neighbor_sum(k, l) - int(adjacent) * a

    return float(epsilon * (a - b) * (neighbors_a - neighbors_b))


# ===== attempt_replica_exchange =====
def attempt_replica_exchange(
    states, energies, replica_ids, temperatures, a, b, rng
):
    """Swap configurations between fixed temperature slots in place.

    states must be a list of independent lattice arrays.
    energies and replica_ids must be one-dimensional NumPy arrays.
    """
    if not isinstance(states, list):
        raise TypeError("states must be a list of lattice arrays.")
    if a == b:
        raise ValueError("Choose two different temperature slots.")

    log_acceptance = replica_exchange_log_acceptance(
        float(energies[a]),
        float(energies[b]),
        float(temperatures[a]),
        float(temperatures[b]),
    )

    if log_acceptance == 0.0:
        accepted = True
    else:
        uniform = float(rng.random())
        accepted = (
            uniform == 0.0 or np.log(uniform) < log_acceptance
        )

    if accepted:
        states[a], states[b] = states[b], states[a]
        energies[[a, b]] = energies[[b, a]]
        replica_ids[[a, b]] = replica_ids[[b, a]]

    return accepted


# ===== replica_exchange_log_acceptance =====
def replica_exchange_log_acceptance(U_a, U_b, T_a, T_b):
    """Log acceptance for a symmetric configuration-swap proposal."""
    values = np.asarray([U_a, U_b, T_a, T_b], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Energies and temperatures must be finite.")
    if T_a <= 0 or T_b <= 0:
        raise ValueError("Temperatures must be positive.")

    log_ratio = (1.0 / T_a - 1.0 / T_b) * (U_a - U_b)
    return float(min(0.0, log_ratio))


# ===== periodic_cluster_observables =====
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


# ===== occupied_wrapping_class =====
def occupied_wrapping_class(components):
    """Classify wrapping using all occupied components."""
    if any(c["winding_rank"] == 2 for c in components):
        return "rank2_present"

    w0 = any(c["wrap_axis0"] for c in components)
    w1 = any(c["wrap_axis1"] for c in components)

    if w0 and w1:
        return "both_projections_rank1"
    if w0:
        return "axis0_only"
    if w1:
        return "axis1_only"
    return "none"

