"""Regression for NPY mapping registration, plus actual full-size CSC pin/sampling."""
import argparse,json,tempfile,time,resource,subprocess,sys
import numpy as np
from candidates.pa_sage_bidir_native_v2.common import *
from candidates.pa_sage_bidir_native_v2.graph_io import load_pinned_csc

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--full-graph',action='store_true');args=parser.parse_args()
    setup();import torch,dgl,runner as r
    r.startup();result=dict(passed=False,raw_ssd_access=False)
    with tempfile.TemporaryDirectory(prefix='digit-csc-map-test-') as tmp:
        path=Path(tmp)
        for name,array in [('indptr',np.array([0,2,4],dtype=np.int64)),('indices',np.array([0,1,0,1],dtype=np.int64)),('eids',np.arange(4,dtype=np.int64))]:
            np.save(path/('original_'+name+'.npy'),array)
        files=list(path.glob('*.npy'));hashes={str(p):sha(p) for p in files}
        # Keep the intentional CUDA registration failure in a fresh process;
        # its pending CUDA error otherwise contaminates a later sampler launch.
        probe = r"""
import sys,json,numpy as np,torch,dgl
from pathlib import Path
p=Path(sys.argv[1])
a=[np.load(p/('original_'+n+'.npy'),mmap_mode='r') for n in ('indptr','indices','eids')]
g=dgl.graph(('csc',tuple(torch.from_numpy(x) for x in a)),num_nodes=2).formats('csc')
try:g._graph.pin_memory_()
except dgl.DGLError as exc:
    assert 'invalid argument' in str(exc)
    print(json.dumps(dict(readonly_registration_error=str(exc))))
else:
    g._graph.unpin_memory_()
    raise RuntimeError('Read-only baseline no longer reproduces')
"""
        process=subprocess.run([sys.executable,'-c',probe,str(path)],capture_output=True,text=True)
        require(process.returncode==0,'Read-only reproduction failed: '+process.stderr)
        result.update(json.loads(process.stdout))
        g,arrays=load_pinned_csc(path,2,4)
        r.seed(0);inp,out,blocks=dgl.dataloading.NeighborSampler([2],replace=False).sample_blocks(g,torch.tensor([0,1],device='cuda'))
        require(blocks[0].num_edges()==4 and sorted(blocks[0].edata[dgl.EID].cpu().tolist())==[0,1,2,3],'Private mapped graph sampling failed')
        g._graph.unpin_memory_()
        arrays[1][0]=1;arrays[1].flush()
        require(np.load(path/'original_indices.npy')[0]==0,'Private write reached source')
        require(all(sha(Path(p))==h for p,h in hashes.items()),'Source file changed')
        result.update(private_mapping_sampling_passed=True,copy_on_write_file_isolation_passed=True)
    if args.full_graph:
        from candidates.pa_sage_direction_v2.graph import verify_sample
        from digit.gpu_admission import check_live
        from candidates.pa_sage_bidir_native_v2.admission import estimate
        admission=check_live(estimate());require(admission['passed'],'Full graph admission failed')
        p=cfg();data=ROOT/p['data'];prepared=read(data/'prepared.json')
        identity=lambda f:(f.stat().st_dev,f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns,f.stat().st_ctime_ns)
        paths=[data/('original_'+name+'.npy') for name in ('indptr','indices','eids')]
        before={str(f):identity(f) for f in paths};start=time.perf_counter()
        g,arrays=load_pinned_csc(data,p['graph']['nodes'],p['graph']['edges'])
        result['full_graph_pin_seconds']=time.perf_counter()-start
        print(json.dumps(dict(stage='full_graph_pinned',seconds=result['full_graph_pin_seconds'])),flush=True)
        require(all(array_sha(a)==p['graph']['csc_sha256'][name] for name,a in zip(('indptr','indices','eids'),arrays)),'Actual pinned arrays differ from selected CSC')
        edges=np.load(source_config()['source_contract']['original_edges']['path'],mmap_mode='r')
        roots=torch.from_numpy(splits('train')[:64].copy()).cuda()
        r.seed(0);inp,out,blocks=dgl.dataloading.NeighborSampler(p['fanouts'],replace=False).sample_blocks(g,roots)
        verify_sample(blocks,edges,p['graph']['nodes'],'bidirectional');torch.cuda.synchronize()
        result.update(full_graph_passed=True,full_graph_nodes=g.num_nodes(),full_graph_edges=g.num_edges(),full_graph_sampled_edges=[b.num_edges() for b in blocks],full_graph_csc_bytes=sum(a.nbytes for a in arrays),pinned_array_sha_matches_selected=True,host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        g._graph.unpin_memory_()
        require(all(identity(f)==before[str(f)] and sha(f)==prepared['bindings'][f.name] for f in paths),'Full source files changed')
        result['source_files_unchanged']=True
    result['passed']=True;write(args.output,result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
