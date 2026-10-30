from pathlib import Path
import urllib.request,json,hashlib,subprocess
H=Path(__file__).resolve().parent;u='https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-77612-y/MediaObjects/41467_2026_77612_MOESM1_ESM.pdf';r={'url':u,'attempt':'conventional publisher supplementary endpoint; identity requires PDF title verification'}
try:
 with urllib.request.urlopen(u,timeout=30) as f:
  length=f.headers.get('Content-Length');r['declared_bytes']=length
  if length and int(length)>20*1024**2:raise RuntimeError('transfer cap exceeded')
  b=f.read(20*1024**2+1)
 assert b.startswith(b'%PDF') and len(b)<=20*1024**2
 p=H/'literature/expTM_published_supplement.pdf';p.write_bytes(b);subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True);r.update(status='downloaded',bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
except Exception as e:r.update(status='unavailable',error=str(e))
(H/'PUBLISHED_SUPPLEMENT_ACCESS.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
