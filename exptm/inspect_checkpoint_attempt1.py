from pathlib import Path
import sys,json,time
import torch
H=Path(__file__).resolve().parent;torch.set_num_threads(2);sys.path.insert(0,str(H/'author_source/GC_Lattice_gas/expTM_data_genenration'))
from tm.architectures.unet_2d_mid_attn import Unet2D
p=H/'author_source/GC_Lattice_gas/expTM_data_genenration/thermomaps-root/models/model_49.pt';v=torch.load(p,map_location='cpu',weights_only=True)
print('keys',list(v));print('epoch',v.get('epoch'));print('model type',type(v['model']));print('optim keys',list(v.get('optim',{})));print('optim states',len(v.get('optim',{}).get('state',{})))
m=Unet2D(dim=32,dim_mults=(1,2,4),resnet_block_groups=8,channels=3);m.load_state_dict(v['model'],strict=True)
r={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'keys':list(v),'epoch':v.get('epoch'),'strict_load':True,'parameters':sum(x.numel() for x in m.parameters()),'tensor_count':len(v['model']),'parameter_shapes':{k:list(x.shape) for k,x in v['model'].items()},'optimizer_states':len(v.get('optim',{}).get('state',{})),'saved_training_configuration':False,'saved_preprocessing':False,'saved_diffusion_schedule':False,'saved_rng_seeds':False}
(H/'CHECKPOINT_INSPECTION.json').write_text(json.dumps(r,indent=2)+'\n');print({k:r[k] for k in ['strict_load','parameters','optimizer_states','saved_training_configuration']})
