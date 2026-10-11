import argparse,gc,math,os,resource,time,shutil
from pathlib import Path
from common import *
from ae.pa_sage.common import setup_imports,source_config
setup_imports()
import numpy as np,torch,dgl
import runner as r
from graph import build,verify_sample
from digit.eval_trace import build_trace,EvaluationTrace
from ae.pa_sage.trace_cache import PreparedEvaluationTrace
from ae.pa_sage.evaluation import rng_fingerprint

class RawPinnedLoader(r.SourceLoader):
 def __init__(self,features,tensor):super().__init__(features);self.tensor=tensor
 def fetch_feature(self,dim,it,device):
  began=time.perf_counter();inp,out,blocks=next(it);self.sample_time+=time.perf_counter()-began
  began=time.perf_counter();x=dgl.utils.gather_pinned_tensor_rows(self.tensor,inp);self.feature_time+=time.perf_counter()-began;self.rows+=len(inp)
  return inp,out,blocks,x

def array_digest(tensor):
 a=tensor.numpy();h=hashlib.sha256();v=memoryview(a).cast('B')
 for off in range(0,len(v),16*1024**2):h.update(v[off:off+16*1024**2])
 return h.hexdigest()

def evaluate(model,loader,labels,trace,limit=None):
 before=rng_fingerprint();mode=model.training;value=r.evaluate(model,loader,None,labels,trace,limit=limit)
 require(before==rng_fingerprint() and model.training==mode,'Evaluation changed training RNG/mode')
 return value

def train(arm,g,edges,features,pinned,labels,splits,traces,output,smoke,p):
 output.mkdir(exist_ok=False);loader=RawPinnedLoader(features,pinned)
 # Remove every smoke side effect from the full training initialization.
 r.seed(p['seed']);model=r.SAGE(128,p['hidden'],p['classes'],num_layers=p['layers'],dropout=p['dropout']).cuda()
 optimizer=torch.optim.Adam(model.parameters(),**p['optimizer']);initial=r.model_hash(model);r.seed(p['seed']);initial_rng=r.dgl_rng();model.train()
 sampler=dgl.dataloading.NeighborSampler(p['fanouts'],replace=False)
 epochs=[];audits=[];total=2 if smoke else p['epochs'];batch=p['batch_size'];train_ids=splits['train'];full_roots=read(ROOT/'data/papers_g2_random_v2/orders.json')['hashes']
 reference=read(ROOT/'results/pa_sage_cache_20260921_v1/cpu10/seed0_repeat0_gids/report.json') if arm=='directed' else None
 if reference:require(initial==reference['initial_parameters_sha256'] and initial_rng==reference['initial_dgl_rng'],'Historical initialization/RNG bridge failed')
 for epoch in range(total):
  start=time.perf_counter();roots=np.random.default_rng(np.random.SeedSequence([p['seed'],epoch])).permutation(train_ids);root_hash=r.digest(roots)
  require(root_hash==full_roots['s0_e%d'%epoch],'Unexpected root order')
  roots_gpu=torch.from_numpy(roots.copy()).cuda();ys=torch.from_numpy(labels[roots].astype(np.int64)).cuda();limit=2 if smoke else math.ceil(len(roots)/batch)
  def batches():
   for i in range(limit):yield sampler.sample_blocks(g,roots_gpu[i*batch:(i+1)*batch])
  it=batches();torch.cuda.synchronize();order_seconds=time.perf_counter()-start;losses=[];counts=0
  for i in range(limit):
   inp,out,blocks,x=loader.fetch_feature(128,it,torch.device('cuda:0'))
   require(torch.equal(out,roots_gpu[i*batch:i*batch+len(out)]),'Sampled root order changed')
   if smoke or i==0:
    require(np.array_equal(x.cpu().numpy().view('u4'),features[inp.cpu().numpy()].view('u4')),'Pinned raw gather differs from source')
    verify_sample(blocks,edges,g.num_nodes(),arm)
   pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,ys[i*batch:i*batch+len(out)])
   optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(loss.detach());counts+=len(out)
   if smoke:audits.append(dict(epoch=epoch,batch=i,feature=r.digest(x),inputs=r.digest(inp),outputs=r.digest(out),loss=float(loss)))
   if (i+1)%100==0 or i+1==limit:progress(output.parent,'smoke_training' if smoke else 'training',arm=arm,epoch=epoch+1,batch=i+1,batches=limit,elapsed_epoch_seconds=time.perf_counter()-start)
  torch.cuda.synchronize();seconds=time.perf_counter()-start;values=torch.stack(losses).cpu().numpy();require(np.isfinite(values).all(),'Nonfinite loss')
  require(all(torch.isfinite(w).all() for w in model.parameters()),'Nonfinite model');require(counts==(2*batch if smoke else len(roots)),'Missing training examples')
  val=evaluate(model,loader,labels,traces['valid'],2 if smoke else None)
  require(val['examples']==(2*batch if smoke else len(splits['valid'])),'Incomplete validation')
  model_hash=r.model_hash(model);bridge=None
  if reference:
   old=reference['epochs'][epoch];bridge=dict(losses_bitwise_equal=np.array_equal(values,np.asarray(old['losses'][:limit],dtype=values.dtype)))
   if not smoke:
    bridge.update(model_hash_equal=model_hash==old['model_sha256'],validation_prediction_hash_equal=val['prediction_sha256']==old['validation']['prediction_sha256'])
    if not all(bridge.values()):
     write(output/'bridge_failure.json',dict(epoch=epoch+1,bridge=bridge,losses=values.tolist(),model_sha256=model_hash,validation=val))
     torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/'bridge_failure_model.pt')
    require(all(bridge.values()),'Directed source-only training differs from historical GIDS; investigate before attributing graph direction')
   elif epoch==0:require(bridge['losses_bitwise_equal'],'First smoke losses differ from historical GIDS')
  rec=dict(epoch=epoch+1,root_sha256=root_hash,updates=limit,examples=counts,train_seconds=seconds,order_seconds=order_seconds,losses=values.tolist(),loss_sha256=r.digest(values),model_sha256=model_hash,validation=val,historical_bridge=bridge)
  epochs.append(rec);write(output/('epoch_%02d.json'%(epoch+1)),rec)
  progress(output.parent,'smoke_epoch_complete' if smoke else 'epoch_complete',arm=arm,epoch=epoch+1,train_seconds=seconds,validation_accuracy=val['accuracy'])
 test=None if smoke else evaluate(model,loader,labels,traces['test'])
 if test is not None:
  require(test['examples']==len(splits['test']),'Incomplete final test')
  if reference:require(test['prediction_sha256']==reference['test']['prediction_sha256'],'Historical final-test bridge failed')
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/'final_model.pt')
 report=dict(passed=True,arm=arm,smoke=smoke,seed=p['seed'],source_only=True,raw_ssd_io=False,initial_parameters_sha256=initial,initial_dgl_rng=initial_rng,final_parameters_sha256=r.model_hash(model),epochs=epochs,test=test,audits=audits,updates=sum(v['updates'] for v in epochs),candidate_sha256=sha(HERE/'manifest.json'),protocol_sha256=sha(PROTOCOL),trace_sha256={k:t.manifest_sha256 for k,t in traces.items()},peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_torch_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
 write(output/'report.json',report);del model,optimizer,sampler,loader;gc.collect();torch.cuda.empty_cache();return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['directed','bidirectional'],required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);a=parser.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);p=read(PROTOCOL);execution=verify();binding=read(a.binding)
 for f in binding['sources'].values():require(identity(f['path'])==f['identity'],'Source file changed')
 require(host()>p['host_admission_gib']*2**30,'Insufficient host available memory')
 r.startup();free,total=torch.cuda.mem_get_info();require(free>=p['gpu_admission_gib']*2**30,'Insufficient free GPU memory')
 cfg=source_config();edges=np.load(cfg['source_contract']['original_edges']['path'],mmap_mode='r');n=cfg['num_nodes'];progress(a.output,'graph_load',arm=a.arm)
 g=build(edges,n,a.arm,notify=lambda **kw:progress(a.output,arm=a.arm,**kw));ip,idx,eid=g.adj_tensors('csc')
 if a.arm=='directed':
  import digit_paths
  base=Path(digit_paths.locations()['papers_csc'])/'csc'
  for got,name in [(ip,'original_indptr.npy'),(idx,'original_indices.npy')]:
   expected=np.load(base/name,mmap_mode='r');require(tuple(got.shape)==expected.shape,'CSC shape mismatch')
   for off in range(0,len(expected),4_000_000):require(np.array_equal(got[off:off+4_000_000].numpy(),expected[off:off+4_000_000]),'Directed CSC differs from frozen graph')
 progress(a.output,'graph_digest',arm=a.arm);graph_info=dict(nodes=n,edges=g.num_edges(),policy=p['graph_policy'],arm=a.arm,csc_sha256={name:array_digest(t) for name,t in [('indptr',ip),('indices',idx),('eids',eid)]})
 write(a.output/'graph.json',graph_info);del ip,idx,eid
 splits={k:r.PRIOR.splits(k) for k in ('train','valid','test')};traces={}
 for split in ('valid','test'):
  progress(a.output,'trace_generation',arm=a.arm,split=split)
  path=a.output/(split+'_trace')
  if a.arm=='directed':
   original=ROOT/'data/papers_g2_random_v2'/(split+'_trace');shutil.copytree(original,path)
   require(sha(path/'manifest.json')==p['reference_bindings'][str(original.relative_to(ROOT)/'manifest.json')],'Directed trace copy changed')
   t=EvaluationTrace(path,verify_files=True)
  else:t=build_trace(g,splits[split],p['fanouts'],p['batch_size'],0,path,source=dict(arm=a.arm,graph=graph_info,protocol_sha256=sha(PROTOCOL),cpu_trace_regeneration_bitwise_guaranteed=False))
  trace=PreparedEvaluationTrace(path,t.manifest_sha256);trace.validate_contract(n,g.num_edges(),p['fanouts'],p['batch_size'],splits[split]);details=trace.prepare(max_bytes=8*2**30);require(details['persistent_cuda_bytes']==0,'Unexpected persistent GPU trace');traces[split]=trace
 require(sum(t.preparation['required_bytes'] for t in traces.values())<=16*2**30,'Combined trace cache exceeds host budget')
 write(a.output/'trace_preparation.json',{k:t.preparation for k,t in traces.items()})
 progress(a.output,'pin_graph_and_raw_features',arm=a.arm)
 g._graph.pin_memory_();features=np.load(cfg['source_features']['path'],mmap_mode='r');pinned=torch.from_numpy(np.asarray(features)).pin_memory();require(pinned.is_pinned(),'Features not pinned')
 # Bind the exact pinned copy including the known NPY header to the raw source hash.
 h=hashlib.sha256()
 with open(cfg['source_features']['path'],'rb') as f:h.update(f.read(features.offset))
 v=memoryview(pinned.numpy()).cast('B')
 for off in range(0,len(v),16*1024**2):h.update(v[off:off+16*1024**2])
 require(h.hexdigest()==cfg['source_features']['sha256'],'Pinned feature copy differs from source');del v
 labels=np.load(cfg['label_identity']['path'],mmap_mode='r').reshape(-1)
 smoke=train(a.arm,g,edges,features,pinned,labels,splits,traces,a.output/'smoke',True,p);require(smoke['passed'],'Smoke failed')
 progress(a.output,'smoke_passed_starting_full',arm=a.arm)
 full=train(a.arm,g,edges,features,pinned,labels,splits,traces,a.output/'full',False,p)
 require(verify()==execution,'Candidate source changed during run')
 for f in binding['sources'].values():require(identity(f['path'])==f['identity'],'Source file changed during run')
 write(a.output/'accepted.json',dict(passed=True,graph_sha256=sha(a.output/'graph.json'),smoke_report_sha256=sha(a.output/'smoke/report.json'),full_report_sha256=sha(a.output/'full/report.json'),checkpoint_sha256=sha(a.output/'full/final_model.pt')))
 progress(a.output,'complete',arm=a.arm,test_accuracy=full['test']['accuracy'])
if __name__=='__main__':main()
