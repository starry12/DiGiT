"""Exercise direction semantics, UVA feature parity and preserved training state."""
import argparse,tempfile
from collections import Counter
from worker import *
from digit.eval_trace import EvaluationTraceError

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();r.startup()
 edges=np.array([[0,1],[0,1],[1,0],[2,1],[3,2],[4,3]],dtype=np.int64);n=8
 features=np.random.default_rng(8).normal(size=(n,128)).astype('float32');pinned=torch.from_numpy(features).pin_memory();labels=np.arange(n,dtype=np.int64);arms={}
 with tempfile.TemporaryDirectory(prefix='digit-direction-test-') as tmp:
  for arm in ('directed','bidirectional'):
   g=build(edges,n,arm);ptr,idx,eids=g.adj_tensors('csc');dest=torch.repeat_interleave(torch.arange(n),ptr[1:]-ptr[:-1]);got=Counter(zip(idx.tolist(),dest.tolist()))
   want=Counter(map(tuple,edges.tolist()));want.update((i,i) for i in range(n))
   if arm=='bidirectional':want.update((v,u) for u,v in edges)
   require(got==want,'Multiedges/reverse/self semantics differ')
   roots=np.arange(n,dtype=np.int64);t=build_trace(g,roots,[3,2,2],4,0,Path(tmp)/arm)
   trace=PreparedEvaluationTrace(Path(tmp)/arm,t.manifest_sha256);trace.validate_contract(n,g.num_edges(),[3,2,2],4,roots);trace.prepare()
   rejected=False
   try:trace.validate_contract(n,g.num_edges()+1,[3,2,2],4,roots)
   except EvaluationTraceError:rejected=True
   require(rejected,'Wrong graph trace was accepted');g._graph.pin_memory_();tracks=[]
   for native_copy in (True,False):
    r.seed(0);model=r.SAGE(128,128,172,num_layers=3,dropout=.2).cuda();opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001);initial=r.model_hash(model);r.seed(0);model.train()
    loader=r.SourceLoader(features) if native_copy else RawPinnedLoader(features,pinned);sampler=dgl.dataloading.NeighborSampler([3,2,2]);losses=[];samples=[]
    for j in range(4):
     item=sampler.sample_blocks(g,torch.tensor([0,2,4,6],device='cuda'));verify_sample(item[2],edges,n,arm)
     inp,out,blocks,x=loader.fetch_feature(128,iter([item]),torch.device('cuda:0'));require(np.array_equal(x.cpu().numpy().view('u4'),features[inp.cpu().numpy()].view('u4')),'Bad raw gather')
     pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,torch.from_numpy(labels[out.cpu().numpy()]).cuda());opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss));samples.append(r.digest(inp))
     ev=evaluate(model,loader,labels,trace)
    tracks.append(dict(initial=initial,losses=losses,inputs=samples,final=r.model_hash(model),predictions=ev['prediction_sha256']))
   require(tracks[0]==tracks[1],'Pinned gather changes training/evaluation versus source CPU copy')
   arms[arm]=dict(edges=g.num_edges(),trajectory=tracks[0],wrong_trace_rejected=True);g._graph.unpin_memory_()
 require(arms['directed']['trajectory']['initial']==arms['bidirectional']['trajectory']['initial'],'Unpaired initialization')
 require(arms['directed']['trajectory']['final']!=arms['bidirectional']['trajectory']['final'],'Direction treatment had no effect in fixture')
 result=dict(passed=True,arms=arms,feature_path_training_parity=True,evaluation_rng_restored=True,raw_ssd_io=False);write(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
