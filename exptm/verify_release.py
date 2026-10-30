"""Read-only hashes, shape and frozen scalar arithmetic; no generation or training."""
from pathlib import Path
import json,hashlib,numpy as np
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for e in json.loads((H/'AUTHOR_TEXT_SOURCE_MANIFEST.json').read_text()):
 p=H/'author_source'/e['path'];b=p.read_bytes();assert len(b)==e['bytes'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['git_blob_sha'] and sha(p)==e['sha256']
for e in json.loads((H/'INPUT_VERIFICATION.json').read_text())+json.loads((H/'COMPATIBLE_CHECKPOINT_VERIFICATION.json').read_text()):
 p=H/'author_source'/e['path'];b=p.read_bytes();assert len(b)==e['bytes'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['git_blob_sha'] and sha(p)==e['sha256']
for name,d in json.loads((H/'PRE_EXECUTION_FREEZE.json').read_text())['files'].items():assert sha(H/name)==d
for stage,n in [('smoke',2),('quantitative',256)]:
 p=H/'outputs'/stage;x=np.load(p/'generated_samples.npy');r=json.loads((p/'RESULTS.json').read_text());rho=(x[:,0]>=0).mean(axis=(1,2));assert x.shape==(n,3,20,20) and np.isfinite(x).all() and np.array_equal(rho,np.load(p/'densities.npy'));assert float(rho.mean())==r['density_mean'];assert np.all(x[:,1]==5) and np.all(x[:,2]==0)
 files=sorted(p.glob('batch_*.npy'));assert np.array_equal(np.concatenate([np.load(f) for f in files]),x)
r=json.loads((H/'QUANTITATIVE_COMPARISON.json').read_text());m=json.loads((H/'RUN_MANIFEST.json').read_text());assert m['quantitative_comparison']==r and m['new_training_updates']==0 and m['protected_Pe_access'] is False and m['successor_executed'] is False
f=H/'RELEASE_FREEZE.json'
if f.exists():
 for p,d in json.loads(f.read_text())['files'].items():assert sha(H/p)==d
print(json.dumps({'PASS':True,'source_files':16,'input_files':5,'registered_quantitative_samples':256,'new_generation':0,'training_updates':0,'published_configuration_verified':False}))
