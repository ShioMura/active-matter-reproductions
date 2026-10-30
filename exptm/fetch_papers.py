from pathlib import Path
import urllib.request,json,hashlib
H=Path(__file__).resolve().parent;(H/'literature').mkdir(exist_ok=True)
urls={'expTM_published_AIP.pdf':'https://www.nature.com/articles/s41467-026-77612-y_reference.pdf','expTM_arxiv_v1.pdf':'https://arxiv.org/pdf/2503.15080v1','expTM_arxiv_v1_supplement.pdf':'https://arxiv.org/src/2503.15080v1/anc/supplement.pdf'}
records=[]
for name,u in urls.items():
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0 (academic research reproduction)'})
  with urllib.request.urlopen(req,timeout=30) as r:b=r.read(20*1024*1024+1)
  assert b.startswith(b'%PDF') and len(b)<=20*1024*1024
  (H/'literature'/name).write_bytes(b);records.append({'name':name,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'status':'downloaded'})
 except Exception as e:records.append({'name':name,'url':u,'status':'unavailable','error':str(e)})
 print(records[-1],flush=True)
(H/'PAPER_ACCESS.json').write_text(json.dumps(records,indent=2)+'\n')
