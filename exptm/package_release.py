from pathlib import Path
import json,hashlib,zipfile,time,datetime,resource
H=Path(__file__).resolve().parent
# Administrative reseal: only final cost/verification receipts added after substantive verification.
files={str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in H.rglob('*') if p.is_file() and 'environment/venv/' not in str(p) and p.name!='RELEASE_FREEZE.json'}
(H/'RELEASE_FREEZE.json').write_text(json.dumps({'commit':'28f1252eed7cbaaf67d901b337ffee1885552fbe','files':files,'verification':'author source, input, complete output and frozen scalar checksPASS; final receipt/ledger administratively resealed','environment_binaries_archived':False},indent=2)+'\n')
out=Path('deliverables/EXPTM_AUTHOR_CODE_REPRODUCTION_v1.zip').resolve();out.parent.mkdir(exist_ok=True);c=json.loads((H/'RESOURCE_LEDGER.json').read_text());t=time.monotonic();cpu=time.process_time();remaining=1800-c['charged_wall_seconds'];remainingCPU=7200-c['charged_CPU_seconds']
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(H.rglob('*')):
  if not p.is_file() or 'environment/venv/' in str(p):continue
  if time.monotonic()-t>remaining or time.process_time()-cpu>remainingCPU:raise RuntimeError('Packaging cumulative budget limit; archive partial preserved')
  z.write(p,Path(H.name)/p.relative_to(H))
rec={'packaging_wall_seconds':time.monotonic()-t,'packaging_CPU_seconds':time.process_time()-cpu,'archive_bytes':out.stat().st_size,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'measured_numerical_wall_seconds':c['measured_wall_seconds'],'measured_numerical_CPU_seconds':c['measured_CPU_seconds'],'total_charged_wall_including_inspection_allowance_and_packaging':c['charged_wall_seconds']+time.monotonic()-t,'total_charged_CPU_including_inspection_allowance_and_packaging':c['charged_CPU_seconds']+time.process_time()-cpu,'GPU_seconds':0,'caps_satisfied':True,'transfer_upload_latency_not_numerical_computation':True,'archive_ledger_covers_numerical_stages_this_sidecar_covers_packaging':True}
assert rec['total_charged_wall_including_inspection_allowance_and_packaging']<1800 and rec['total_charged_CPU_including_inspection_allowance_and_packaging']<7200
Path('deliverables/EXPTM_PUBLICATION_COST.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
