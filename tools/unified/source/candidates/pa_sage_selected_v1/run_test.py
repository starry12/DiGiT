"""Thin single-test controller; frozen selection is never changed after inference."""
import argparse,fcntl,shutil,subprocess,signal,traceback
from common import *
from ae.pa_sage.common import source_config
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.full_validation import checkpoint_hash

def gpu_state(gpu):
 out=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip();uuid,used,free=[v.strip() for v in out.split(',')]
 return dict(uuid=uuid,used_bytes=int(used)*2**20,free_bytes=int(free)*2**20)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',default='2');a=parser.parse_args();a.output=a.output.resolve();require(not (HERE/'test_invocation.json').exists(),'This frozen checkpoint already has a test invocation; do not rerun');a.output.mkdir(parents=True,exist_ok=False)
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['PYTHONDONTWRITEBYTECODE']='1';state=dict(passed=False,stage='starting',pid=os.getpid(),started_unix=time.time(),gpu=a.gpu,test_evaluations=0);child=None;monitor=None
 def status(stage,**kw):state.update(stage=stage,updated_unix=time.time(),**kw);write(a.output/'status.json',state);print(json.dumps(state),flush=True)
 def stopped(signum,frame):raise KeyboardInterrupt('Controller signaled')
 signal.signal(signal.SIGTERM,stopped)
 try:
  lock=open('/tmp/digit-pa-direction.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execution=verify();cfg=read(CONFIG);before=gpu_state(a.gpu)
  require(before['used_bytes']<2**30 and before['free_bytes']>cfg['gpu_admission_gib']*2**30,'Selected GPU busy/too small');require(host()>cfg['host_admission_gib']*2**30,'Insufficient host memory');require(shutil.disk_usage(a.output).free>2*2**30,'Insufficient output space')
  write(a.output/'admission.json',dict(passed=True,gpu=before,host_available_bytes=host(),freeze_manifest_sha256=execution));status('synthetic_validation')
  with (a.output/'synthetic.log').open('x') as log:
   child=subprocess.Popen([sys.executable,'-u',str(HERE/'test_worker.py'),'--synthetic','--output',str(a.output/'synthetic.json')],stdout=log,stderr=subprocess.STDOUT);code=child.wait();child=None;require(code==0,'Single-pass synthetic validation failed')
  source=source_config();sources={}
  for name,d in dict(features=source['source_features'],labels=source['label_identity'],split_test=source['splits']['test']).items():
   status('source_hash',source=name);s=identity(d['path']);digest=sha(d['path']);require(digest==d['sha256'] and s==identity(d['path']),'Source hash/identity mismatch');sources[name]=dict(path=d['path'],sha256=digest,identity=s)
  write(a.output/'sources.json',dict(sources=sources,freeze_manifest_sha256=execution));monitor=ExternalMonitor(a.output/'gpu_monitor');monitor.start()
  command=[sys.executable,'-u',str(HERE/'test_worker.py'),'--output',str(a.output/'worker'),'--binding',str(a.output/'sources.json')];write(a.output/'command.json',dict(command=command,CUDA_VISIBLE_DEVICES=a.gpu))
  with (a.output/'worker.log').open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);status('running',worker_pid=child.pid);receipt=a.output/'worker/test_complete.json';monitor_info=None;deadline=None
   while child.poll() is None:
    if receipt.exists() and monitor_info is None:
     require(read(receipt)['passed'],'Worker failed test acceptance');monitor_code=monitor.stop();monitor_info=read(a.output/'gpu_monitor/summary.json');monitor_info['returncode']=monitor_code;monitor=None
     write(a.output/'release_worker.json',dict(test_complete_sha256=sha(receipt),monitor_stopped=True,monitor_passed=monitor_info['passed'],time_unix=time.time()));status('releasing_worker',test_evaluations=1,monitor_passed=monitor_info['passed']);deadline=time.monotonic()+180
    if deadline is not None:require(time.monotonic()<deadline,'Worker exit timed out')
    time.sleep(.5)
   code=child.wait();child=None;require(code==0 and receipt.exists() and monitor_info is not None,'Test worker did not complete; see worker.log')
  status('verifying_test',worker_returncode=code);report=read(a.output/'worker/report.json');require(report['passed'] and sha(a.output/'worker/report.json')==read(receipt)['report_sha256'],'Test report binding mismatch')
  require(report['test_evaluations']==1 and report['validation_evaluations']==report['optimizer_updates']==0,'Wrong evaluation/training extent')
  require(report['test']['examples']==cfg['test_examples'] and report['test']['batches']==report['forward_calls']==cfg['test_batches'],'Incomplete full test')
  require(report['checkpoint_epoch']==20 and report['checkpoint_sha256']==cfg['checkpoint_sha256'] and report['model_sha256']==checkpoint_hash(HERE/cfg['checkpoint_file'])==cfg['model_sha256'],'Epoch20 checkpoint changed')
  require(verify()==execution==report['freeze_manifest_sha256'],'Frozen selection changed')
  for f in sources.values():require(identity(f['path'])==f['identity'],'Source changed after evaluation')
  import numpy as np
  pred=np.load(a.output/'worker/test_predictions.npy');ids=np.load(a.output/'worker/test_nodes.npy');labels=np.load(source['label_identity']['path'],mmap_mode='r').reshape(-1);target=labels[ids].astype(np.int64)
  require(sha(a.output/'worker/test_predictions.npy')==report['predictions_file_sha256'] and sha(a.output/'worker/test_nodes.npy')==report['test_nodes_file_sha256'],'Saved prediction/root evidence changed')
  require(len(pred)==len(ids)==cfg['test_examples'] and len(np.unique(ids))==len(ids) and report['test']['accuracy']==float((pred==target).mean()),'Independent saved-prediction accuracy mismatch')
  summary=dict(passed=True,test_evaluations=1,validation_evaluations=0,optimizer_updates=0,seed=cfg['seed'],checkpoint_epoch=20,selected_case=cfg['selected_case'],optimizer=cfg['optimizer'],selection_metrics=cfg['selection_metrics'],checkpoint_sha256=cfg['checkpoint_sha256'],model_sha256=cfg['model_sha256'],freeze_manifest_sha256=execution,trace_manifest_sha256=cfg['test_trace_sha256'],test=report['test'],worker_returncode=code,monitor_passed=monitor_info['passed'],monitor_errors=monitor_info['errors'],source_only=True,raw_ssd_io=False,performance_claim=False,saved_predictions_independently_verified=True,gpu_after=gpu_state(a.gpu));write(a.output/'summary.json',summary)
  lines=['# 已冻结配置与epoch20 checkpoint：一次完整test完成','', '**配置：双向 PA / GraphSAGE，Adam lr=0.001、weight_decay=0，seed0，第20轮checkpoint。**','', '| 指标 | 结果 |','|---|---:|','| 最后5轮validation均值（选择依据） | %.4f%% |'%(100*cfg['selection_metrics']['last5_mean_validation']),'| epoch20 validation | %.4f%% |'%(100*cfg['selection_metrics']['epoch20_validation']),'| 完整test | **%.4f%%** |'%(100*report['test']['accuracy']),'| test样本数 / batch数 | %d / %d |'%(cfg['test_examples'],cfg['test_batches']),'', 'test恰好一次完整前向遍历；没有额外训练、validation或换checkpoint。配置与权重先冻结，再启动test；防重入凭据位于冻结目录test_invocation.json。模型权重/RNG在评估前后不变，test roots逐批匹配官方split；保存了逐样本预测与节点ID，控制器据此重新计算准确率。','', 'worker exit 0；monitor passed='+str(monitor_info['passed'])+'，errors='+str(len(monitor_info['errors']))+'；监控先结束再释放大块锁页内存。','', '本结果是按论文重建的原始特征路径、单seed选定模型test结果。它不代表双向图原生GIDS/DiGiT配对验收、性能复现或多seed收敛；历史模型已有的test结果不被视为本次重复运行。','', '[冻结配置](../../candidates/pa_sage_selected_v1/config.json) · [冻结manifest](../../candidates/pa_sage_selected_v1/manifest.json) · [epoch20权重](../../candidates/pa_sage_selected_v1/final_model_epoch20.pt) · [完整报告](summary.json) · [预测](worker/test_predictions.npy) · [节点ID](worker/test_nodes.npy) · [状态](status.json)']
  (a.output/'README.md').write_text('\n'.join(lines)+'\n');status('complete' if monitor_info['passed'] else 'complete_with_monitor_warning',passed=True,test_evaluations=1,worker_returncode=0,monitor_passed=monitor_info['passed'],finished_unix=time.time(),summary_sha256=sha(a.output/'summary.json'))
 except BaseException as exc:
  status('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise
 finally:
  if monitor is not None:monitor.stop()
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=90)
   except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
