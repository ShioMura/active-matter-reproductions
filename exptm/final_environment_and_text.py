from pathlib import Path
import importlib.metadata as md,json,sys,platform,subprocess,torch
H=Path(__file__).resolve().parent
for p in (H/'literature').glob('*.pdf'):subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
names=['numpy','scipy','pandas','scikit-learn','PyYAML','matplotlib','torch','einops','tqdm'];v={n:md.version(n) for n in names}
(H/'environment/used_versions.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':v,'torch_CUDA_available':torch.cuda.is_available(),'CPU_threads':2,'author_versions_not_pinned':True},indent=2)+'\n')
(H/'environment/execution_requirements.txt').write_text('\n'.join(f'{n}=={ver}' for n,ver in v.items() if n!='torch')+'\n')
(H/'environment/pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True));print(v)
