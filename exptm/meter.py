"""Cumulative active-computation guard. CPU only; no stage rerun without a new attempt label."""
from pathlib import Path
import os,sys,json,time,subprocess,resource,signal,datetime
H=Path(__file__).resolve().parent;ledger=H/'RESOURCE_LEDGER.json'
def run(label,cmd):
 d=json.loads(ledger.read_text()) if ledger.exists() else {'limits':{'active_wall_seconds':1800,'CPU_seconds':7200,'GPU_seconds':1200},'GPU_available':False,'GPU_seconds':0,'inspection_allowance_not_measured':{'wall_seconds':15,'CPU_seconds':15},'stages':[]}
 assert not any(s['label']==label for s in d['stages']),'Existing attempt label; no hidden retry'
 wall=1800-15-sum(x['wall_seconds'] for x in d['stages']);cpu=7200-15-sum(x['CPU_seconds'] for x in d['stages']);assert wall>0 and cpu>0
 def limit():resource.setrlimit(resource.RLIMIT_CPU,(int(cpu)+1,int(cpu)+2))
 t=time.monotonic();own=time.process_time();r0=resource.getrusage(resource.RUSAGE_CHILDREN);env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
 with (H/f'{label}.stdout.log').open('w') as o,(H/f'{label}.stderr.log').open('w') as e:
  p=subprocess.Popen(cmd,stdout=o,stderr=e,env=env,start_new_session=True,preexec_fn=limit);terminated=False;peakcpu=0
  while p.poll() is None:
   try:
    stat=Path(f'/proc/{p.pid}/stat').read_text().split();peakcpu=max(peakcpu,(int(stat[13])+int(stat[14])+int(stat[15])+int(stat[16]))/os.sysconf('SC_CLK_TCK'))
   except FileNotFoundError:pass
   if time.monotonic()-t>=wall-2 or peakcpu>=cpu-2:
    terminated=True;os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=1)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    break
   time.sleep(.5)
  p.wait()
 r=resource.getrusage(resource.RUSAGE_CHILDREN);d['stages'].append({'label':label,'command':cmd,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-t,'CPU_seconds':r.ru_utime+r.ru_stime-r0.ru_utime-r0.ru_stime+time.process_time()-own,'max_child_RSS_KiB':r.ru_maxrss,'returncode':p.returncode,'budget_terminated':terminated});d['measured_wall_seconds']=sum(x['wall_seconds'] for x in d['stages']);d['measured_CPU_seconds']=sum(x['CPU_seconds'] for x in d['stages']);d['charged_wall_seconds']=d['measured_wall_seconds']+15;d['charged_CPU_seconds']=d['measured_CPU_seconds']+15;ledger.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stages'][-1]));return p.returncode
if __name__=='__main__':sys.exit(run(sys.argv[1],sys.argv[2:]))
