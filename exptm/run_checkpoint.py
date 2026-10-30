"""One literal author lattice-gas recipe; only CPU checkpoint/path adapter, no author source edits."""
from pathlib import Path
import argparse,json,sys,hashlib,time,random,platform
import numpy as np
import torch
H=Path(__file__).resolve().parent;SRC=H/'author_source/GC_Lattice_gas/expTM_data_genenration';sys.path.insert(0,str(SRC));torch.set_num_threads(2);torch.set_num_interop_threads(1)
from tm.core.loader import Loader
from tm.core.prior import GlobalEquilibriumHarmonicPrior
from tm.architectures.unet_2d_mid_attn import Unet2D
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_process import VPDiffusion
from tm.core.diffusion_model import SteeredDiffusionSampler
C=json.loads((H/'FROZEN_TARGET.json').read_text())
def setup():
 random.seed(C['seeds']['prepare']);np.random.seed(C['seeds']['prepare']);torch.manual_seed(C['seeds']['prepare'])
 low=np.load(H/'author_source/GC_Lattice_gas/data/LowMuT7.npy');high=np.load(H/'author_source/GC_Lattice_gas/data/HighMuT7.npy');low[:,1]=-4;high[:,1]=4
 data=np.concatenate((low,high));data=np.concatenate((data,np.ones((len(data),1,20,20))),axis=1);data[:,1]=5;pressure=np.concatenate((-np.ones(len(low)),np.ones(len(high))));data[:,2]=pressure[:,None,None];temperature=np.full(len(data),5.)
 loader=Loader(data=data,pressures=pressure[:,None],temperatures=temperature[:,None],control_dims=(1,3))
 prior=GlobalEquilibriumHarmonicPrior(shape=loader.data.shape,channels_info={'coordinate':[0],'fluctuation':[1],'mean':[2]})
 model=Unet2D(dim=32,dim_mults=(1,2,4),resnet_block_groups=8,channels=3)
 bb=ConvBackbone(model=model,data_shape=loader.data_dim,target_shape=32,num_dims=4,lr=1e-3,eval_mode='train',self_condition=True)
 # Public loading interface with an explicit available device. No checkpoint resizing/key dropping.
 bb.load_model(str(SRC/'thermomaps-root/models'),1,device='cpu')
 dp=VPDiffusion(num_diffusion_timesteps=100)
 sampler=SteeredDiffusionSampler(dp,bb,loader,prior,pred_type='noise')
 steps=[]
 for v in bb.optim.state_dict()['state'].values():
  if 'step' in v:steps.append(float(v['step']))
 meta={'loader_shape':list(loader.data.shape),'prior_shape':prior.shape,'temperature_channel':5,'normalized_mu_values':[-1,1],'backbone_target_shape':32,'checkpoint_epoch':bb.start_epoch-1,'parameters':sum(p.numel() for p in model.parameters()),'optimizer_step_values':sorted(set(steps)),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'platform':platform.platform(),'normal_transform_unused_for_author_decode':True}
 return sampler,loader,meta

def main(stage):
 p=H/'outputs'/stage;p.mkdir(parents=True,exist_ok=False);sampler,loader,meta=setup();seed=C['seeds'][stage];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);n=C['smoke_samples'] if stage=='smoke' else C['quantitative_samples'];torch.save(torch.get_rng_state(),p/'initial_sampling_rng.pt');np.savez_compressed(p/'loader_transforms.npz',mean=loader.transform.mean.numpy(),std=loader.transform.std.numpy());t=time.monotonic();completed=0;parts=[];batch=2 if stage=='smoke' else 16
 for i in range(n//batch):
  start=time.monotonic();x=sampler.sample_loop(num_samples=batch,batch_size=batch,temperature=5,pressure=0,eta=1,gamma=1);a=x.detach().cpu().numpy();assert a.shape==(batch,3,20,20) and np.isfinite(a).all();assert np.all(a[:,1]==5) and np.all(a[:,2]==0)
  np.save(p/f'batch_{i:03d}.npy',a);torch.save(torch.get_rng_state(),p/'current_sampling_rng.pt');completed+=batch;parts.append(a)
  (p/'progress.json').write_text(json.dumps({'completed_samples':completed,'planned_samples':n,'elapsed_sampling_wall_seconds':time.monotonic()-t,'last_batch_wall_seconds':time.monotonic()-start},indent=2));print(json.dumps({'stage':stage,'completed':completed,'batch_wall':time.monotonic()-start}),flush=True)
 a=np.concatenate(parts);rho=(a[:,0]>=0).mean(axis=(1,2));np.save(p/'densities.npy',rho);np.save(p/'generated_samples.npy',a)
 result=dict(meta,stage=stage,seed=seed,samples=n,batch_size=batch,sampling_wall_seconds=time.monotonic()-t,finite=True,shape=list(a.shape),density_mean=float(rho.mean()),density_sd=float(rho.std(ddof=1)),descriptive_conditional_SEM=float(rho.std(ddof=1)/np.sqrt(n)),fixed_physical_reference=.5,absolute_difference=float(abs(rho.mean()-.5)),Hoeffding95_halfwidth=float(np.sqrt(np.log(40)/(2*n))),density_min=float(rho.min()),density_max=float(rho.max()),threshold_decode_zero_inclusive=True,independent_training=False,published_configuration_verified=False)
 result['Hoeffding95_interval']=[max(0,result['density_mean']-result['Hoeffding95_halfwidth']),min(1,result['density_mean']+result['Hoeffding95_halfwidth'])];result['exact_reference_inside_conditional_bound']=result['Hoeffding95_interval'][0]<=.5<=result['Hoeffding95_interval'][1]
 (p/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(torch.get_rng_state(),p/'final_sampling_rng.pt');print(json.dumps(result))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','quantitative']);main(parser.parse_args().stage)
