"""Bounded implementation checks; no production MC or equilibrium decision."""
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
import numpy as np
import pt_rotation_runner as r


def run_checks():
    results = []
    config = r.default_config()
    b = r.budget(config)
    assert b["within_slot_proposals"] == 1_440_000_000
    assert b["swap_attempts"] == 3_150_000
    assert b["segments"] == 90
    for kind in config["seeds"]:
        state = r.make_state(kind, config, np.random.default_rng(123))
        assert state.sum() == 200
        shape = r.dep.periodic_cluster_observables(state)
        if kind == "droplet":
            assert r.dep.occupied_wrapping_class(shape["components"]) == "none"
            assert shape["f_max"] == 1.0
        if kind == "stripe":
            assert r.dep.occupied_wrapping_class(shape["components"]) == "axis0_only"
            np.testing.assert_allclose(r.observables(state, r.direct_energy(state)), [-380, 20.4317290945307, -40.8634581890614, 40.8634581890614, 0], atol=1e-10)
        original = r.observables(state, r.direct_energy(state))
        for k in range(4):
            changed = np.rot90(state, k).copy()
            measured = r.observables(changed, r.direct_energy(changed))
            np.testing.assert_allclose(np.array(measured)[:4], [original[0], original[1], original[2]*(-1)**k, original[3]], atol=1e-10)
    results.append("20x20 initialization, generic FFT normalization, rotation invariants")

    # Exhaustive fixed-N 3x3 configurations, all distinct ordered site pairs.
    energy_cases = 0
    for occupied in itertools.combinations(range(9), 4):
        state = np.zeros((3, 3), dtype=np.int8)
        state.flat[list(occupied)] = 1
        old = r.direct_energy(state)
        for a in range(9):
            for b in range(9):
                if a == b:
                    continue
                i, j = divmod(a, 3)
                k, l = divmod(b, 3)
                predicted = r.dep.nonlocal_swap_energy_change(state, i, j, k, l)
                new = state.copy()
                new[i,j], new[k,l] = state[k,l], state[i,j]
                assert r.direct_energy(new) - old == predicted
                energy_cases += 1
    results.append(f"{energy_cases} exhaustive 3x3 N=4 energy-difference checks")

    test = copy.deepcopy(config)
    test.update(protocol="IMPLEMENTATION_TEST_ONLY", L=4, N=8, temperatures=[.4,.5,.7,1.],
                cycles=12, initial_exclusion=4, checkpoint_every=4, snapshot_every=2,
                check_every=2, progress_every=100, seeds={"random":[9042], "stripe":[9043], "droplet":[9044]})
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        continuous = r._create(tmp, test)
        resumed = r._create(tmp, test)
        assert r.run_campaign(continuous)["status"].endswith("ASSESSMENT_PENDING")
        assert r.run_campaign(resumed, max_new_segments=1)["status"] == "PAUSED"
        r.run_campaign(resumed)
        interrupted = r._create(tmp, test)
        original_cycle = r.dep.replica_exchange_cycle
        counter = [0]
        def fail_mid_segment(*args, **kwargs):
            counter[0] += 1
            if counter[0] == 7:
                raise KeyboardInterrupt("Injected mid-segment interruption")
            return original_cycle(*args, **kwargs)
        r.dep.replica_exchange_cycle = fail_mid_segment
        try:
            r.run_campaign(interrupted)
            raise AssertionError("Interruption was not injected")
        except KeyboardInterrupt:
            pass
        finally:
            r.dep.replica_exchange_cycle = original_cycle
        assert not (interrupted / "RUNNING.lock").exists()
        r.run_campaign(interrupted)
        keys_checked = 0
        for baseline in continuous.glob("*/*.npz"):
            relative = baseline.relative_to(continuous)
            expected = r.load_npz(baseline)
            for folder in (resumed, interrupted):
                observed = r.load_npz(folder / relative)
                assert expected.keys() == observed.keys()
                for key in expected:
                    if key == "meta_json":  # timings and predecessor container hashes differ
                        continue
                    np.testing.assert_array_equal(expected[key], observed[key], err_msg=str(relative)+":"+key)
                    keys_checked += 1
        results.append(f"Continuous / checkpoint pause / mid-segment interrupt: {keys_checked} array-and-RNG comparisons identical")
        # A completed call is idempotent and does not perform more updates.
        before = {str(p): r.digest(p) for p in resumed.glob("*/*.npz")}
        r.run_campaign(resumed)
        assert before == {str(p): r.digest(p) for p in resumed.glob("*/*.npz")}
        results.append("Completed campaign rerun does not alter trajectories")
        # Explicit lock and frozen-protocol protections.
        (resumed / "RUNNING.lock").write_text("test", encoding="utf-8")
        try:
            r.run_campaign(resumed)
            raise AssertionError("Concurrent execution was not blocked")
        except RuntimeError:
            pass
        (resumed / "RUNNING.lock").unlink()
        path = resumed / "protocol.json"
        manifest = json.loads(path.read_text())
        manifest["config"]["cycles"] = 16
        path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            r.read_manifest(resumed)
            raise AssertionError("Changed protocol was not rejected")
        except ValueError:
            pass
        # Remove an interior segment: a discontinuous history must not resume.
        (interrupted / "random_9042" / "segment_000004_000008.npz").unlink()
        try:
            r.run_campaign(interrupted)
            raise AssertionError("Missing segment was not rejected")
        except ValueError:
            pass
        results.append("Concurrent lock, protocol modification, and missing segment rejected")
    return {"status": "IMPLEMENTATION_CHECKS_PASSED", "checks": results,
            "environment": r.environment(), "source_sha256": r.code_hashes(),
            "scope": "Small implementation tests only; no 20x20 production trajectory or equilibrium assessment."}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
