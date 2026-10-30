from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
for name,h in json.loads((H/'PRE_EXECUTION_FREEZE.json').read_text())['files'].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==h
p=H/'outputs/quantitative';r=json.loads((p/'RESULTS.json').read_text());x=np.load(p/'generated_samples.npy');rho=np.load(p/'densities.npy');assert x.shape==(256,3,20,20) and np.array_equal(rho,(x[:,0]>=0).mean(axis=(1,2)))
with (H/'DENSITY_COMPARISON.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['sample_index','decoded_occupancy_density','exact_symmetry_reference','temperature_code_units','normalized_mean_control']);w.writerows([[i,float(d),.5,5,0] for i,d in enumerate(rho)])
# All draws retained. Fixedfirst6 display, fixed[0,1] density axis; no outcome-selected examples.
fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist(rho,bins=np.linspace(0,1,21));ax[0].axvline(.5,c='k',ls='--',label='exact symmetry reference (conditional mapping)');ax[0].axvline(rho.mean(),c='tab:red',label='checkpoint mean');ax[0].set(xlim=(0,1),xlabel='decoded density per configuration',ylabel='generated sample count');ax[0].legend(fontsize=7)
lo,hi=r['Hoeffding95_interval'];ax[1].errorbar([0],[r['density_mean']],yerr=[[r['density_mean']-lo],[hi-r['density_mean']]],fmt='o',capsize=8,label='conditional Hoeffding95% enclosure');ax[1].axhline(.5,c='k',ls='--');ax[1].set(ylim=(0,1),xticks=[0],xticklabels=['model_1 / nativeT5 / mean-control0'],ylabel='mean decoded density');ax[1].legend(fontsize=7);fig.tight_layout();fig.savefig(H/'density_comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(8,5))
for i,a in enumerate(axs.ravel()):a.imshow(x[i,0]>=0,vmin=0,vmax=1,cmap='Greys');a.set_title(f'draw{i}, rho={rho[i]:.4f}');a.set_xticks([]);a.set_yticks([])
fig.tight_layout();fig.savefig(H/'first_six_generated_lattices.png',dpi=160);plt.close(fig)
r['unique_raw_configuration_hashes']=len({hashlib.sha256(s.tobytes()).hexdigest() for s in x});r['unique_thresholded_configurations']=len({hashlib.sha256((s[0]>=0).tobytes()).hexdigest() for s in x});r['plots_show_all_densities_and_first_six_fixed_indices']=True
(H/'QUANTITATIVE_COMPARISON.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['density_mean','density_sd','absolute_difference','Hoeffding95_interval','exact_reference_inside_conditional_bound','unique_thresholded_configurations']}))
