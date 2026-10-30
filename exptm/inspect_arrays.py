from pathlib import Path
import numpy as np,json
h=Path(__file__).resolve().parent;res={}
for name in ['HighMuT7.npy','LowMuT7.npy','GC_density_heatmap.npy']:
 p=h/'author_source/GC_Lattice_gas/data'/name;x=np.load(p,mmap_mode='r');print(name,x.shape,x.dtype)
 v={'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max())}
 if x.ndim==4:
  v['channels']=[{'min':float(x[:,k].min()),'max':float(x[:,k].max()),'mean':float(x[:,k].mean()),'unique_first_frame':np.unique(x[0,k]).tolist()} for k in range(x.shape[1])]
 else:v['values']=x.tolist()
 res[name]=v;print(v)
(h/'ARRAY_INSPECTION.json').write_text(json.dumps(res,indent=2)+'\n')
