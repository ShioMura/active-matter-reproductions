"""Frozen, finite-time PEP map. No training or stationary phase inference."""
from pathlib import Path
import hashlib, json, time, platform, uuid
from datetime import datetime, timezone
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pep_core import make_state, assert_state, pep_sweep, measure, occupancy, orientation_field, CHECK_RESULTS

CONFIG = {
    'task': 'PEP_FINITE_TIME_MAP_11_V1', 'L': 20,
    'densities': np.linspace(0.1, 0.9, 11).tolist(),
    'persistence_A': np.geomspace(1.0, 20.0, 11).tolist(),
    'seeds': [2026111141, 2026111142], 'sweeps': 500,
    'measure_every': 10, 'summary_sweeps': list(range(410, 501, 10)),
    'local_window': 5, 'initial': 'random',
    'rng': 'SeedSequence([seed, density_index, stream]); stream 0=init, 1=updates; paired across A',
    'question': 'At a fixed observation time, is stronger persistence associated with larger density inhomogeneity across densities?',
    'descriptive_comparison': 'For each density report A=20 minus A=1 for each seed and both observables; positive in both seeds is descriptive agreement only.',
    'technical_success': '242 complete conserved trajectories; finite arrays; all original implementation checks pass; saved snapshots reproduce measurements.',
    'stop': 'Stop on check failure or after 242 trajectories. No seed selection, extension, tuning, training, or stationary-phase claim.',
    'stationarity_validated': False, 'training_data_qualified': False,
}
FROZEN = json.dumps(CONFIG, sort_keys=True, separators=(',', ':'))
PROTOCOL_SHA = hashlib.sha256(FROZEN.encode()).hexdigest()
OBS = ['S_low', 'local_density_variance', 'f_max', 'move_fraction']

def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)

def run_map(root):
    assert json.dumps(CONFIG, sort_keys=True, separators=(',', ':')) == FROZEN
    assert CHECK_RESULTS['status'] == 'IMPLEMENTATION_CHECKS_PASSED'
    root = Path(root)
    folder = root / ('map11_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    # Freeze protocol, seeds, source, and decision rules before any simulation.
    write_json(folder/'protocol.json', CONFIG)
    write_json(folder/'implementation_checks.json', CHECK_RESULTS)
    write_json(folder/'provenance.json', {
        'protocol_sha256': PROTOCOL_SHA,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('pep_core.py')]},
        'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'model': 'Soto and Golestanian PRE 89,012706 (2014), Section II; PEP, no attraction',
        'parent_decisions': 'Earlier equilibrium NOT READY and Track B STOP remain unchanged.'})
    rows = []
    started = time.perf_counter()
    attempts_total = 0
    write_json(folder/'status.json', {'status':'running', 'completed_trajectories':0})
    try:
        for di, rho in enumerate(CONFIG['densities']):
            for ai, A in enumerate(CONFIG['persistence_A']):
                alpha = 1/A
                for si, seed in enumerate(CONFIG['seeds']):
                    tag = f'd{di:02d}_a{ai:02d}_s{si}'
                    out = folder/tag
                    out.mkdir()
                    init_rng = np.random.default_rng(np.random.SeedSequence([seed, di, 0]))
                    rng = np.random.default_rng(np.random.SeedSequence([seed, di, 1]))
                    state = make_state(CONFIG['L'], rho, init_rng)
                    times, occupations, orientations = [0], [occupancy(state)], [orientation_field(state)]
                    measurements = [dict(sweep=0, move_fraction=None, **measure(state, CONFIG['local_window']))]
                    moved = attempts = 0
                    for sweep in range(1, CONFIG['sweeps']+1):
                        m, a = pep_sweep(state, alpha, rng)
                        moved += m; attempts += a; attempts_total += a
                        if sweep % CONFIG['measure_every'] == 0:
                            assert_state(state)
                            measurements.append(dict(sweep=sweep, move_fraction=moved/attempts, **measure(state, CONFIG['local_window'])))
                            times.append(sweep); occupations.append(occupancy(state)); orientations.append(orientation_field(state))
                            moved = attempts = 0
                        if sweep % 100 == 0:
                            np.savez_compressed(out/'checkpoint.npz', owner=state['owner'], positions=state['positions'], orientation=state['orientation'])
                            write_json(out/'checkpoint.json', {'sweep':sweep, 'L':state['L'], 'N':state['N'], 'rng_state':rng.bit_generator.state})
                            write_json(out/'measurements.json', measurements)
                    np.savez_compressed(out/'snapshots.npz', sweeps=times, occupation=occupations, orientation=orientations)
                    window = [r for r in measurements if r['sweep'] in CONFIG['summary_sweeps']]
                    assert len(window) == 10
                    record = dict(tag=tag, density_index=di, activity_index=ai, seed_index=si,
                                  density=rho, N=state['N'], A=A, alpha=alpha, seed=seed,
                                  window_mean={k:float(np.mean([r[k] for r in window])) for k in OBS})
                    assert all(np.isfinite(v) for v in record['window_mean'].values())
                    rows.append(record)
                    write_json(folder/'summary.json', rows)
                    write_json(folder/'status.json', {'status':'running', 'completed_trajectories':len(rows), 'elapsed_seconds':time.perf_counter()-started})
            print(f'Density {di+1}/11 complete | trajectories {len(rows)}/242 | {time.perf_counter()-started:.1f}s', flush=True)
        assert len(rows) == 242 and attempts_total == 24200000
        write_json(folder/'status.json', {'status':'completed', 'completed_trajectories':len(rows),
                  'attempts':attempts_total, 'seconds':time.perf_counter()-started,
                  'stationarity_validated':False, 'training_data_qualified':False})
    except BaseException as exc:
        write_json(folder/'status.json', {'status':'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                  'completed_trajectories':len(rows), 'error':repr(exc)})
        raise
    return folder

def load_arrays(folder):
    folder = Path(folder)
    config = json.loads((folder/'protocol.json').read_text())
    assert json.dumps(config, sort_keys=True, separators=(',', ':')) == FROZEN
    assert json.loads((folder/'status.json').read_text())['status']=='completed'
    rows = json.loads((folder/'summary.json').read_text())
    arrays = {k:np.full((2,11,11), np.nan) for k in OBS}
    for r in rows:
        for k in OBS:
            arrays[k][r['seed_index'],r['density_index'],r['activity_index']] = r['window_mean'][k]
    assert all(np.isfinite(a).all() for a in arrays.values())
    return config, arrays

def draw_map(ax, values, title, config, vmin, vmax, cmap='viridis'):
    A = np.array(config['persistence_A']); rho = np.array(config['densities'])
    ax.set_xscale('log')
    im = ax.pcolormesh(A, rho, values, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(A[[0,2,4,6,8,10]], [f'{x:.2g}' for x in A[[0,2,4,6,8,10]]])
    ax.set_xlabel('Persistence A = 1/alpha (log scale)'); ax.set_ylabel('Density rho')
    ax.set_title(title, fontsize=11)
    return im

def plot_results(folder):
    folder = Path(folder)
    config, arrays = load_arrays(folder)
    np.savez_compressed(folder/'map_arrays.npz', A=config['persistence_A'], alpha=1/np.array(config['persistence_A']),
                        density=config['densities'], seeds=config['seeds'], **arrays)
    labels = {'S_low':'Low-wavevector structure S_low', 'local_density_variance':'Local-density variance (5 x 5 window)'}
    fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    for ax,k in zip(axes,labels):
        avg = arrays[k].mean(axis=0)
        im = draw_map(ax,avg,labels[k],config,0,float(avg.max()))
        fig.colorbar(im,ax=ax,label='Mean over two seed-level window means')
    fig.suptitle('PEP 11 x 11 finite-time clustering map | L=20\nSweeps 410–500; 2 seeds per point; not a stationary phase diagram',fontsize=13)
    fig.savefig(folder/'clustering_map.png',dpi=160); plt.show()

    fig, axes = plt.subplots(2,3, figsize=(15,8), layout='constrained')
    for row,k in enumerate(labels):
        a=arrays[k]; vmax=float(a.max())
        for si in range(2):
            im=draw_map(axes[row,si],a[si],f'{k} | seed {config["seeds"][si]}',config,0,vmax)
            fig.colorbar(im,ax=axes[row,si])
        difference=np.abs(a[0]-a[1])
        im=draw_map(axes[row,2],difference,'Absolute difference of seed-level means',config,0,max(float(difference.max()),1e-12),'magma')
        fig.colorbar(im,ax=axes[row,2])
    fig.suptitle('Seed variability | same colour scale for the two seeds within each row\nTwo seeds do not provide a reliable confidence interval',fontsize=13)
    fig.savefig(folder/'seed_comparison.png',dpi=150); plt.show()

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,k in zip(axes,labels):
        difference=(arrays[k]-arrays[k][:,:,0,None]).mean(axis=0)
        bound=max(float(np.abs(difference).max()),1e-12)
        im=draw_map(ax,difference,f'{k}: difference from matched-density A=1',config,-bound,bound,'RdBu_r')
        fig.colorbar(im,ax=ax,label='Paired mean difference (descriptive)')
    fig.suptitle('Density-matched SEP control | finite-time window; no significance test',fontsize=13)
    fig.savefig(folder/'matched_density_contrast.png',dpi=160); plt.show()
    endpoint=[]
    for di,rho in enumerate(config['densities']):
        record={'density':rho}
        for k in labels:
            diff=arrays[k][:,di,-1]-arrays[k][:,di,0]
            record[k]={'A20_minus_A1_by_seed':diff.tolist(),'positive_in_both_seeds':bool(np.all(diff>0))}
        endpoint.append(record)
    write_json(folder/'endpoint_comparison.json',endpoint)
    return arrays

def verify_saved(folder):
    """Recompute both plotted observables from all retained measurement frames."""
    from pep_core import lowest_mode_structure, local_density_field
    folder=Path(folder)
    rows=json.loads((folder/'summary.json').read_text())
    frames=0
    for r in rows:
        snap=np.load(folder/r['tag']/'snapshots.npz',allow_pickle=False)
        measurements=json.loads((folder/r['tag']/'measurements.json').read_text())
        assert len(measurements)==len(snap['sweeps'])==51
        for t,n,o,m in zip(snap['sweeps'],snap['occupation'],snap['orientation'],measurements):
            assert int(t)==m['sweep'] and n.sum()==r['N']
            assert np.array_equal(o>=0,n.astype(bool))
            assert np.isclose(lowest_mode_structure(n),m['S_low'],rtol=0,atol=1e-12)
            assert np.isclose(local_density_field(n,5).var(),m['local_density_variance'],rtol=0,atol=1e-12)
            frames+=1
        window=[m for m in measurements if m['sweep'] in CONFIG['summary_sweeps']]
        for k in OBS: assert np.isclose(np.mean([m[k] for m in window]),r['window_mean'][k],rtol=0,atol=1e-12)
    write_json(folder/'validation.json',{'status':'SAVED_DATA_CHECKS_PASSED','trajectories':len(rows),'frames_recomputed':frames,
               'checks':'Conservation, occupation/orientation agreement, both heatmap observables recomputed, fixed-window means verified.'})
    return frames

if __name__=='__main__':
    folder=run_map(Path(__file__).parent/'runs')
    print('RUN_FOLDER',folder,flush=True)
    plot_results(folder)
    print('RECOMPUTED_FRAMES',verify_saved(folder),flush=True)
