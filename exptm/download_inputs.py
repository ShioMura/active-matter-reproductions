from pathlib import Path
import urllib.request,json,hashlib,time
H=Path(__file__).resolve().parent;m=json.loads((H/'INPUT_TRANSFER_MANIFEST.json').read_text());out=[]
for item in m['files']:
 p=H/'author_source'/item['path'];p.parent.mkdir(parents=True,exist_ok=True);url='https://raw.githubusercontent.com/tiwarylab/ExpTM/'+m['commit']+'/'+item['path'];start=time.monotonic()
 if not p.exists():
  with urllib.request.urlopen(url,timeout=120) as r,p.with_suffix(p.suffix+'.partial').open('wb') as f:
   while True:
    b=r.read(1024*1024)
    if not b:break
    f.write(b)
  p.with_suffix(p.suffix+'.partial').replace(p)
 b=p.read_bytes();assert len(b)==item['size'];gh=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert gh==item['sha'];out.append(dict(path=item['path'],bytes=len(b),git_blob_sha=gh,sha256=hashlib.sha256(b).hexdigest(),source_url=url,download_wall_seconds=time.monotonic()-start));print(out[-1],flush=True)
(H/'INPUT_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
