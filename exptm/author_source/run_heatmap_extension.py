"""Bounded public-checkpoint control-grid diagnostic. No training or physical-axis claim."""
import argparse, hashlib, json, os, platform, random, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Match this CPU diagnostic even on a GPU host.
import numpy as np
import torch

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def read_cell(path, i, j, t, p, n):
    with np.load(path, allow_pickle=False) as saved:
        a, rho = saved['samples'], saved['densities']
        assert a.shape == (n,3,20,20) and np.isfinite(a).all(), path
        assert np.all(a[:,1] == np.float32(t)) and np.all(a[:,2] == np.float32(p)), path
        np.testing.assert_array_equal(rho, (a[:,0]>=0).mean(axis=(1,2)))
        wall = float(saved['wall_seconds'])
    return dict(i=i,j=j,T_control=t,p_control=p,n=n,mean=float(rho.mean()),
        sem=float(rho.std(ddof=1)/np.sqrt(n)),empty=int((rho==0).sum()),
        full=int((rho==1).sum()),wall_seconds=wall)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Existing reproduction project root')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--resume', action='store_true', help='Validate config and continue missing cells only')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    src = root / 'author_source/GC_Lattice_gas/expTM_data_genenration'
    ckpt = src / 'thermomaps-root/models/model_1.pt'
    expected = '72125a26b5d136b8eeda782ff01fbecb99963b3ad58b8fce3f4494185f5419fc'
    assert digest(ckpt) == expected, 'Unexpected public checkpoint'
    sys.path.insert(0, str(src))
    from tm.core.loader import Loader
    from tm.core.prior import GlobalEquilibriumHarmonicPrior
    from tm.core.backbone import ConvBackbone
    from tm.core.diffusion_model import SteeredDiffusionSampler
    from tm.core.diffusion_process import VPDiffusion
    from tm.architectures.unet_2d_mid_attn import Unet2D
    import tm.core.diffusion_model as module
    assert Path(module.__file__).resolve() == src / 'tm/core/diffusion_model.py'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    temperatures = [2.0, 6.0, 10.0]
    pressures = [-2.0, -1.5, 1.5, 2.0]
    shape = (len(temperatures), len(pressures))
    total = shape[0] * shape[1]
    seed, n = 2026103101, 16
    data_dir = root / 'author_source/GC_Lattice_gas/data'
    data_files = [data_dir / k for k in ['LowMuT7.npy', 'HighMuT7.npy']]
    config = dict(purpose='Preliminary Figure 2(b)-style attempt in code-control coordinates',
        physical_parameter_mapping_verified=False, figure_checkpoint_identity_verified=False,
        temperature_controls=temperatures, p_controls=pressures, samples_per_cell=n,
        batch_size=n, sampling_seed_reset_each_cell=seed, setup_seed=42,
        shared_random_numbers_across_cells=True, grid_cell_errors_not_independent=True,
        checkpoint_sha256=expected, checkpoint_file=str(ckpt),
        data_sha256={p.name:digest(p) for p in data_files},
        source_sha256={str(p.relative_to(src)):digest(p) for p in sorted((src/'tm').rglob('*.py'))},
        diffusion_steps=100, eta=1, gamma=1, backbone_eval_mode='train',
        decoder='coordinate channel >= 0', training=False, device='cpu',
        python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
        runner_sha256=digest(Path(__file__).resolve()),
        stopping_rule='Exactly 12 cells; stop on invalid outputs or after 6000 cumulative successful-cell sampling seconds between cells; no adaptive extension',
        success_definition='Finite correctly shaped samples and a labelled control-grid plot; not physical reproduction success')
    if args.resume:
        old = json.loads((out/'config.json').read_text(encoding='utf-8'))
        assert old == config, 'Config/source/data/environment changed. Resume refused; keep the original run.'
    else:
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out/'config.json',config)
    rows=[]
    for i,t in enumerate(temperatures):
        for j,p in enumerate(pressures):
            path=out/f'cell_{i}_{j}.npz'
            if path.is_file(): rows.append(read_cell(path,i,j,t,p,n))
    done={(r['i'],r['j']) for r in rows}
    print(f'Validated {len(done)}/{total} existing cells; {total-len(done)} remaining.',flush=True)
    if len(done)==total:
        means=np.empty(shape); sems=np.empty(shape)
        for r in rows:means[r['i'],r['j']]=r['mean']; sems[r['i'],r['j']]=r['sem']
        atomic_json(out/'results.json',rows)
        np.savez(out/'heatmap.npz',mean=means,sem=sems,T=temperatures,p=pressures)
        if not (out/'completion.json').exists():
            atomic_json(out/'completion.json',dict(status='CONTROL_GRID_COMPLETED',samples=total*n,
                sampling_wall_seconds=sum(r['wall_seconds'] for r in rows),physical_reproduction_validated=False))
        print('All cells already complete. No new sampling.',flush=True)
        return
    spent=sum(r['wall_seconds'] for r in rows)
    if spent>=6000:raise RuntimeError('Frozen cumulative sampling-time cap reached.')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    low, high = [np.load(p, allow_pickle=False) for p in data_files]
    assert low.shape == high.shape == (5000,2,20,20)
    p_labels = np.repeat([-1.,1.], [len(low),len(high)])
    data = np.zeros((len(p_labels),3,20,20), dtype=np.float64)
    data[:,0] = np.concatenate([low[:,0],high[:,0]])
    data[:,1] = 5.
    data[:,2] = p_labels[:,None,None]
    loader = Loader(data=data, pressures=p_labels[:,None], temperatures=np.full((len(data),1),5.), control_dims=(1,3))
    prior = GlobalEquilibriumHarmonicPrior(shape=loader.data.shape, channels_info={'coordinate':[0],'fluctuation':[1],'mean':[2]})
    model = Unet2D(dim=32,dim_mults=(1,2,4),resnet_block_groups=8,channels=3)
    bb = ConvBackbone(model=model,data_shape=loader.data_dim,target_shape=32,num_dims=4,lr=1e-3,eval_mode='train',self_condition=True)
    bb.load_model(str(ckpt.parent),1,device='cpu')
    sampler = SteeredDiffusionSampler(VPDiffusion(num_diffusion_timesteps=100),bb,loader,prior,pred_type='noise')
    means = np.full(shape,np.nan); sems = means.copy()
    for r in rows: means[r['i'],r['j']]=r['mean']; sems[r['i'],r['j']]=r['sem']
    started=time.monotonic()
    for i,t in enumerate(temperatures):
        for j,p in enumerate(pressures):
            if (i,j) in done:continue
            if spent > 6000:
                raise RuntimeError('Frozen wall-time cap reached; retain partial results')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            cell_start=time.monotonic()
            with torch.no_grad():
                a=sampler.sample_loop(num_samples=n,batch_size=n,temperature=t,pressure=p,eta=1,gamma=1).detach().cpu().numpy()
            assert a.shape == (n,3,20,20) and np.isfinite(a).all()
            assert np.all(a[:,1] == np.float32(t)) and np.all(a[:,2] == np.float32(p))
            rho=(a[:,0]>=0).mean(axis=(1,2))
            means[i,j]=rho.mean(); sems[i,j]=rho.std(ddof=1)/np.sqrt(n)
            wall=time.monotonic()-cell_start
            # Atomic cell files make interruption/resume safe. Each cell resets its RNG.
            temp_path=out/f'cell_{i}_{j}.tmp.npz'
            np.savez_compressed(temp_path,samples=a,densities=rho,wall_seconds=wall)
            temp_path.replace(out/f'cell_{i}_{j}.npz')
            row=dict(i=i,j=j,T_control=t,p_control=p,n=n,mean=float(means[i,j]),sem=float(sems[i,j]),
                empty=int((rho==0).sum()),full=int((rho==1).sum()),wall_seconds=wall)
            rows.append(row)
            spent+=wall
            rows.sort(key=lambda r:(r['i'],r['j']))
            atomic_json(out/'results.json',rows)
            print(json.dumps(dict(completed=len(rows),total=total,estimated_remaining_minutes=(spent/len(rows))*(total-len(rows))/60,**row)),flush=True)
    np.savez(out/'heatmap.npz',mean=means,sem=sems,T=temperatures,p=pressures)
    atomic_json(out/'completion.json',dict(status='CONTROL_GRID_COMPLETED',samples=total*n,
        sampling_wall_seconds=spent,current_session_wall_seconds=time.monotonic()-started,physical_reproduction_validated=False))

if __name__ == '__main__':
    main()
