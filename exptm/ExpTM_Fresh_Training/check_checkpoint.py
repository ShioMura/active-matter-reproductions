"""Read-only, fixed-size diagnostic; no optimizer updates or full sampling."""
from pathlib import Path
import argparse, hashlib, json, sys, time, platform
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
SEED=2026110902
TIMES=[0,24,49,74,89,98,99]
N=16
protocol=dict(question='At which fixed forward-noising levels does this checkpoint produce large clean-coordinate reconstruction errors?',
 seed=SEED,samples_per_anchor=N,timesteps=TIMES,controls=[-1.0,1.0],temperature_control=1.0,
 sample_source='Seeded selection from supplied training arrays; NOT held-out validation',
 dequantization='Normal noise, scale 0.01, applied to all three channels as in Loader',
 comparison='Constant per-pixel training mean reconstruction, converted to a noise predictor',
 stopping='One fixed evaluation; no training, optimizer updates, parameter changes or full reverse trajectories',
 interpretation='Descriptive diagnostic, no scientific PASS threshold or causal attribution',
 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
 python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform())
(args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
source=args.project/'author_training_source'
manifest=json.loads((args.project/'training_manifest.json').read_text())
for relative,sha in manifest['sha256'].items():
 assert hashlib.sha256((args.project/relative).read_bytes()).hexdigest()==sha,relative
sys.path.insert(0,str(source))
from tm.architectures.unet_2d_mid_attn import Unet2D
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_process import VPDiffusion
from tm.core.prior import GlobalEquilibriumHarmonicPrior
torch.set_num_threads(2)
torch.manual_seed(SEED)
model=Unet2D(dim=32,dim_mults=(1,2,4),resnet_block_groups=8,channels=3)
bb=ConvBackbone(model=model,data_shape=20,target_shape=32,num_dims=4,
                lr=1e-3,eval_mode='train',self_condition=True)
assert bb.device=='cpu'
state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
model.load_state_dict(state['model'],strict=True)
assert int(state['epoch'])==4
model.eval()
def weight_digest():
 h=hashlib.sha256()
 for name,tensor in model.state_dict().items():
  h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
 return h.hexdigest()
before=weight_digest()
dp=VPDiffusion(num_diffusion_timesteps=100)
prior=GlobalEquilibriumHarmonicPrior(shape=(N,3,20,20),channels_info={'coordinate':[0],'fluctuation':[1],'mean':[2]})
rows=[];initial_rows=[];saved={}
started=time.perf_counter()
rng=np.random.default_rng(SEED)
with torch.no_grad():
 for p,filename in [(-1.0,'LowMuT7.npy'),(1.0,'HighMuT7.npy')]:
  data=np.load(args.project/'data'/filename,allow_pickle=False)
  idx=rng.choice(len(data),N,replace=False)
  clean=torch.empty(N,3,20,20)
  clean[:,0]=torch.from_numpy(data[idx,0]).float();clean[:,1]=1;clean[:,2]=p
  x0=clean+torch.randn_like(clean)*.01
  eps=prior.sample(batch_size=N,temperatures=np.ones((N,1)),pressures=np.full((N,1),p))
  clean_mean=torch.empty(1,3,20,20)
  clean_mean[:,0]=torch.from_numpy(data[:,0].mean(0)).float()
  clean_mean[:,1]=1;clean_mean[:,2]=p
  saved[f'indices_{p:+.0f}']=idx
  saved[f'x0_{p:+.0f}']=x0.numpy()
  saved[f'epsilon_{p:+.0f}']=eps.numpy()
  for t in TIMES:
   times=torch.full((N,),t,dtype=torch.long);a=dp.alphas[t]
   xt=a.sqrt()*x0+(1-a).sqrt()*eps
   reconstructed,predicted=dp.reverse_kernel(xt,times,bb,'noise')
   assert torch.isfinite(reconstructed).all()
   d=predicted[:,0]-eps[:,0];e=reconstructed[:,0]-x0[:,0]
   baseline=(xt-a.sqrt()*clean_mean)/(1-a).sqrt()
   baseline_rmse=(baseline[:,0]-eps[:,0]).square().mean().sqrt()
   oracle=(xt-(1-a).sqrt()*eps)/a.sqrt()
   multiplier=((1-a)/a).sqrt()
   algebra_error=(e+multiplier*d).abs().max()
   row=dict(mean_control=p,t=t,alpha=float(a),error_multiplier=float(multiplier),
     coordinate_noise_rmse=float(d.square().mean().sqrt()),coordinate_noise_bias=float(d.mean()),
     coordinate_reconstruction_rmse=float(e.square().mean().sqrt()),
     reconstructed_coordinate_median=float(reconstructed[:,0].median()),
     reconstructed_coordinate_mean=float(reconstructed[:,0].mean()),
     reconstructed_density=float((reconstructed[:,0]>=0).float().mean()),
     clean_selected_density=float((clean[:,0]>=0).float().mean()),
     constant_mean_noise_rmse=float(baseline_rmse),
     realized_noise_algebra_recovery_max_error=float((oracle-x0).abs().max()),
     error_identity_max_residual=float(algebra_error))
   rows.append(row)
   if t in [0,98,99]:saved[f'reconstructed_{p:+.0f}_t{t}']=reconstructed.numpy()
  # Exactly the sampler's terminal initialization, using the same fixed prior noise.
  times=torch.full((N,),99,dtype=torch.long)
  xhat,pred=dp.reverse_kernel(eps,times,bb,'noise')
  initial_rows.append(dict(mean_control=p,t=99,
    reconstructed_coordinate_median=float(xhat[:,0].median()),
    reconstructed_coordinate_mean=float(xhat[:,0].mean()),
    reconstructed_coordinate_rms=float(xhat[:,0].square().mean().sqrt()),
    reconstructed_density=float((xhat[:,0]>=0).float().mean())))
  saved[f'prior_initialized_xhat_{p:+.0f}']=xhat.numpy()
after=weight_digest()
assert before==after
result=dict(checkpoint_epoch=int(state['epoch']),strict_state_dict_load=True,weights_unchanged=True,
 model_self_condition=model.self_condition,backbone_self_condition=bb.self_condition,
 elapsed_seconds=time.perf_counter()-started,rows=rows,prior_initialization=initial_rows,
 limitations=['Training examples reused; not an independent physical test.',
 'PyTorch 2.5.1 CPU here versus recorded 2.14.1 CPU in the user run.',
 'Single-step reconstruction at noisy known inputs is not a full reverse trajectory.',
 'Realized-noise algebra recovery is not an implementable reverse-time predictor.',
 'Does not prove that longer training will fix the mismatch or identify its sole cause.'])
(args.output/'results.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(args.output/'diagnostic_arrays.npz',**saved)
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for p in [-1.,1.]:
 r=[x for x in rows if x['mean_control']==p];ts=[x['t'] for x in r]
 axes[0].semilogy(ts,[x['coordinate_noise_rmse'] for x in r],'o-',label=f'Model, p={p:+.0f}')
 axes[0].semilogy(ts,[x['constant_mean_noise_rmse'] for x in r],'--',label=f'Constant mean, p={p:+.0f}')
 axes[1].semilogy(ts,[x['coordinate_reconstruction_rmse'] for x in r],'o-',label=f'p={p:+.0f}')
axes[2].semilogy(TIMES,[x['error_multiplier'] for x in rows[:len(TIMES)]],'o-',color='black')
for ax,title in zip(axes,['Coordinate noise prediction RMSE','Clean-coordinate reconstruction RMSE','Error amplification factor']):
 ax.set_title(title);ax.set_xlabel('Forward diffusion index t');ax.grid(alpha=.2)
axes[0].legend(fontsize=7);axes[1].legend()
fig.tight_layout();fig.savefig(args.output/'checkpoint_diagnostic.png',dpi=160)
print(json.dumps(result,indent=2))
