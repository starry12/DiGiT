"""Thin controller: source binding, independent monitoring, sequential paired runs."""
import argparse,fcntl,shutil,signal,subprocess,traceback
from common import *
from ae.pa_sage.common import source_config
from ae.pa_sage.monitor_control import ExternalMonitor

def gpu_state(gpu):
 result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
 uuid,used,free=[s.strip() for s in result.stdout.strip().split(',')]
 return dict(uuid=uuid,used_bytes=int(used)*2**20,free_bytes=int(free)*2**20)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',default='2');a=parser.parse_args();a.output=a.output.resolve();require(not a.output.exists(),'Use a fresh output directory')
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['PYTHONDONTWRITEBYTECODE']='1'
 a.output.mkdir(parents=True);state=dict(passed=False,stage='starting',pid=os.getpid(),started_unix=time.time(),gpu=a.gpu,raw_ssd_io=False,arms={});monitor=None;child=None
 def status(stage,**kw):state.update(stage=stage,updated_unix=time.time(),**kw);write(a.output/'status.json',state);print(json.dumps(state),flush=True)
 def stop(signum,frame):raise KeyboardInterrupt('Controller received signal '+str(signum))
 signal.signal(signal.SIGTERM,stop)
 try:
  lock=open('/tmp/digit-pa-direction.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  execution=verify();p=read(PROTOCOL);cfg=source_config();before=gpu_state(a.gpu)
  require(before['used_bytes']<1024**3,'Selected GPU is busy');require(before['free_bytes']>p['gpu_admission_gib']*2**30,'Insufficient GPU memory');require(host()>p['host_admission_gib']*2**30,'Insufficient host memory');require(shutil.disk_usage(a.output).free>p['disk_admission_gib']*2**30,'Insufficient output disk space')
  write(a.output/'admission.json',dict(gpu=before,host_available_bytes=host(),disk_free_bytes=shutil.disk_usage(a.output).free,protocol=p,executable=sys.executable,candidate_sha256=execution))
  # Run the actual CUDA feature/graph tests in a separate small process.
  status('synthetic_validation')
  with (a.output/'tests.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-u',str(HERE/'tests.py'),'--output',str(a.output/'tests.json')],stdout=log,stderr=subprocess.STDOUT);require(child.wait()==0,'Synthetic validation failed; see tests.log');child=None
  sources={};specs=dict(features=cfg['source_features'],edges=cfg['source_contract']['original_edges'],labels=cfg['label_identity'])
  specs.update({'split_'+k:v for k,v in cfg['splits'].items()})
  for name,desc in specs.items():
   status('source_hash',source=name);prior=identity(desc['path']);digest=sha(desc['path']);require(digest==desc['sha256'] and identity(desc['path'])==prior,'Raw source identity/hash changed: '+name);sources[name]=dict(path=desc['path'],identity=prior,sha256=digest)
  reference=ROOT/'results/pa_sage_cache_20260921_v1/cpu10/seed0_repeat0_gids/report.json'
  sources['directed_reference']=dict(path=str(reference),identity=identity(reference),sha256=sha(reference))
  write(a.output/'sources.json',dict(sources=sources,candidate_sha256=execution,protocol_sha256=sha(PROTOCOL)))
  monitor=ExternalMonitor(a.output/'gpu_monitor');monitor.start();reports={}
  for arm in p['arms']:
   require(verify()==execution,'Candidate changed');require(gpu_state(a.gpu)['used_bytes']<1024**3,'Selected GPU acquired by another process')
   command=[sys.executable,'-u',str(HERE/'worker.py'),'--arm',arm,'--output',str(a.output/arm),'--binding',str(a.output/'sources.json')]
   state['arms'][arm]=dict(started_unix=time.time(),command=command);status('running',active_arm=arm)
   with (a.output/(arm+'.log')).open('x') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state['arms'][arm]['pid']=child.pid;status('running');code=child.wait();child=None
   state['arms'][arm].update(returncode=code,finished_unix=time.time());require(code==0,'Worker failed: '+arm+'; see '+arm+'.log')
   accepted=read(a.output/arm/'accepted.json');report=read(a.output/arm/'full/report.json');require(accepted['passed'] and report['passed'],'Worker did not accept results')
   require(sha(a.output/arm/'full/report.json')==accepted['full_report_sha256'],'Report binding mismatch');require(sha(a.output/arm/'full/final_model.pt')==accepted['checkpoint_sha256'],'Checkpoint binding mismatch')
   require(len(report['epochs'])==20 and report['updates']==23580 and report['test']['examples']==214338,'Incomplete experiment')
   require(all(e['examples']==1207179 and e['updates']==1179 and e['validation']['examples']==125265 and e['validation']['training_rng_restored'] for e in report['epochs']),'Incomplete train/validation or RNG restoration')
   reports[arm]=report;status('arm_complete')
  left,right=(reports[k] for k in p['arms']);require(left['initial_parameters_sha256']==right['initial_parameters_sha256'] and left['initial_dgl_rng']==right['initial_dgl_rng'],'Unpaired initialization')
  require([e['root_sha256'] for e in left['epochs']]==[e['root_sha256'] for e in right['epochs']],'Unpaired orders')
  require(all(all(e['historical_bridge'].values()) for e in left['epochs']),'Directed historical bridge failed')
  require(verify()==execution,'Candidate changed during execution')
  for name,f in sources.items():require(identity(f['path'])==f['identity'],'Source changed during execution: '+name)
  code=monitor.stop();require(code==0 and read(a.output/'gpu_monitor/summary.json')['passed'],'Monitor failed');monitor=None
  after=gpu_state(a.gpu)
  summary=dict(passed=True,seed=0,epochs=20,candidate_sha256=execution,source_binding_sha256=sha(a.output/'sources.json'),paired_initialization=True,paired_root_orders=True,directed_historical_trajectory_bitwise_equal=True,raw_ssd_io=False,performance_claim=False,accuracy={k:dict(final_validation=v['epochs'][-1]['validation']['accuracy'],best_validation=max(e['validation']['accuracy'] for e in v['epochs']),final_test=v['test']['accuracy']) for k,v in reports.items()},test_difference_percentage_points=100*(right['test']['accuracy']-left['test']['accuracy']),gpu_after=after,limitations=['Single seed; cannot establish multi-seed convergence','Reverse-edge augmentation preserves original multiedges; does not recover the unavailable paper configuration','Source-only accuracy diagnostic; no SSD or GIDS-DiGiT performance conclusion'])
  write(a.output/'summary.json',summary)
  lines=['# PA/SAGE 有向图与双向图受控对照','', '已完成：原始特征路径，seed 0，各 20 epoch，每轮完整 validation，最后一次完整 test。','', '| 图 | epoch 20 valid | best valid（未用于选择 test 模型） | epoch 20 test |','|---|---:|---:|---:|']
  for name,v in summary['accuracy'].items():lines.append('| %s | %.4f%% | %.4f%% | %.4f%% |'%(name,100*v['final_validation'],100*v['best_validation'],100*v['final_test']))
  lines+=['','双向图 test 相对有向图变化：%+.4f 个百分点。'%summary['test_difference_percentage_points'],'','有向组逐轮 loss、模型哈希、validation 预测及最终 test 预测与历史 GIDS 精确一致。两组仅改变反向边增强，原图重复边保留，自环每节点一条。','', '本结果只用于精度归因，单 seed；不能用这里的时间解释 SSD 或 GIDS–DiGiT 性能，也不证明已恢复论文原始配置。','', '协议与入口：`'+str(HERE)+'`；完整证据见 `summary.json`、各组 `full/report.json` 和 `accepted.json`。']
  (a.output/'README.md').write_text('\n'.join(lines)+'\n');status('complete',passed=True,finished_unix=time.time(),summary_sha256=sha(a.output/'summary.json'))
 except BaseException as exc:
  status('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  if monitor is not None:monitor.stop()
if __name__=='__main__':main()
