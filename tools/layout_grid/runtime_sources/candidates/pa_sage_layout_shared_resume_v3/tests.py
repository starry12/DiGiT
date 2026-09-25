"""CPU checks for metadata-only layouts, physical-row features and calibration."""
import argparse,copy,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
from .common import ROOT,HERE,Path,read,write,sha,require,setup

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='digit-shared-pool-test-');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_all_metadata_only_layouts_and_shared_model_updates(self):
        import torch,dgl,runner as r
        from candidates.pa_sage_layout_sweep_v1.tests import fixture
        from candidates.pa_sage_layout_sweep_v1.common import grid
        from .overlay import build_overlay
        from .build import Context,build_layout
        from .bundle import load_metadata
        from .settings import sampler_kwargs,loader_kwargs,create_model
        from .audit import feature_audit
        from candidates.pa_sage_bidir_native_v2.overlay import apply
        from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW
        torch.set_num_threads(1);dgl.utils.set_num_threads(1)
        inputs,edges,bidir,reverse=fixture(self.root/'inputs')
        bp=np.load(bidir/'original_indptr.npy');bi=np.load(bidir/'original_indices.npy')
        graph=dgl.graph(('csc',(torch.from_numpy(bp),torch.from_numpy(bi),torch.empty(0,dtype=torch.int64))),num_nodes=96)
        initial=None;records=[]
        pool=np.random.default_rng(9).normal(0,.05,size=(2048,128)).astype('float32')
        for cell in grid():
            base=self.root/cell['id'];ctx=Context(cell,inputs,96,edges,8,base,fixture=True)
            b=build_layout(ctx);self.assertFalse((base/'final/bundle/reordered_features.npy').exists())
            self.assertFalse(b['filesystem_features_bit_exact']);self.assertTrue(b['metadata_only'])
            validation=read(base/'final/validation.json')
            if cell['id']=='g2_r20':self.assertTrue(validation['passed'])
            else:self.assertIsNone(validation['passed']);self.assertEqual(validation['status'],'skipped_by_user')
            build_overlay(base,inputs['indptr']['path'],bidir,reverse,base/'overlay',fixture=True)
            bundle=load_metadata(base);bundle=apply(bundle,np.load(base/'overlay/reordered_indptr.npy'),np.load(base/'overlay/reordered_indices.npy'),len(bi))
            p=read(ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/protocols'/(cell['id']+'.json'))
            p=copy.deepcopy(p);p['graph'].update(nodes=96,edges=len(bi));p['cpu_cache_rows']=8
            sampler=DiGiTNeighborSampler(p['fanouts'],bundle,**dict(sampler_kwargs(p,bundle.manifest),cuda_mode='disabled'))
            kw=loader_kwargs(p,bundle.manifest,412316860416,bundle.io_geometry)
            self.assertEqual(kw['num_ele'],bundle.manifest['feature']['num_storage_rows']*128)
            self.assertEqual(kw['off'],412316860416)
            def seed_cpu(v):np.random.seed(v);torch.manual_seed(v);dgl.seed(v)
            model,optim,model0=create_model(p,0,'cpu',seed_cpu,r.SAGE,torch,r.model_hash)
            if initial is None:initial=model0
            self.assertEqual(initial,model0)
            audits=[]
            for i in range(2):
                inp,out,blocks=sampler.sample_blocks(graph,torch.arange(i*8,(i+1)*8,dtype=torch.int64))
                rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].numpy();self.assertLess(int(rows.max()),len(pool))
                x=torch.from_numpy(pool[rows].copy())
                # Runner audit normally records CUDA RNG. Replace only this
                # diagnostic for a CPU fixture, not feature/address checks.
                with patch.object(torch.cuda,'get_rng_state',return_value=torch.get_rng_state()):
                    audits.append(feature_audit(inp,out,blocks,x,pool,bundle,'shared'))
                    bad=x.clone();bad[0,0]+=1
                    with self.assertRaises(AssertionError):feature_audit(inp,out,blocks,bad,pool,bundle,'shared')
                pred=model(blocks,x);loss=torch.nn.functional.cross_entropy(pred,out%172)
                optim.zero_grad(set_to_none=True);loss.backward();self.assertTrue(torch.isfinite(loss));optim.step()
            self.assertNotEqual(model0,r.model_hash(model))
            records.append(dict(point=cell,metadata_only=True,updates=2,physical_row_feature_audit=True))
        self.__class__.records=records
        self.assertFalse(torch.cuda.is_initialized())

    def test_protocols_keep_controlled_fields_and_pool_binding(self):
        from .protocol import validate
        index=read(ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/protocols/index.json')
        self.assertEqual(len(index['points']),15)
        for item in index['points']:
            p=read(ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/protocols'/(item['point']['id']+'.json'));validate(p)
            for key,value in [('epochs',2),('feature_mode','fake'),('pool_spec_sha256','wrong')]:
                q=copy.deepcopy(p);q[key]=value
                with self.assertRaises(RuntimeError):validate(q)

    def test_real_budget_cli_import_order(self):
        p=ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/protocols/g2_r20.json';output=self.root/'budget.json'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DIGIT_LAYOUT_PROTOCOL='/invalid/inherited.json')
        result=subprocess.run([sys.executable,'-B','-m','candidates.pa_sage_layout_shared_resume_v3.cli',
               '--protocol',str(p),'--budget-output',str(output)],cwd=ROOT,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(read(output)['protocol_sha256'],sha(p))

    def test_only_pilot_keeps_smoke(self):
        from .run import worker_modes
        from .common import verification_policy
        from candidates.pa_sage_layout_sweep_v1.common import grid
        for cell in grid():
            modes=worker_modes(dict(verification_policy=verification_policy(cell)))
            self.assertEqual(modes,('smoke','full') if cell['id']=='g2_r20' else ('full',))

    def test_graph_binding_uses_container_hash_not_array_hash(self):
        import hashlib
        from .graph_binding import bind,descriptors
        arrays=dict(indptr=np.array([0,1,2],dtype='<i8'),indices=np.array([0,1],dtype='<i8'),eids=np.array([0,1],dtype='<i8'))
        prepared={};graph=dict(nodes=2,edges=2,csc_sha256={})
        for name,a in arrays.items():
            file='original_'+name+'.npy';np.save(self.root/file,a)
            raw=hashlib.sha256(a.tobytes()).hexdigest();container=sha(self.root/file)
            self.assertNotEqual(raw,container)
            prepared[name]=dict(file=file,sha256=container,array_sha256=raw,shape=list(a.shape));graph['csc_sha256'][name]=raw
        receipt=dict(passed=True,validation=dict(passed=True,nodes=2,edges=2),graph=prepared,bindings={d['file']:d['sha256'] for d in prepared.values()})
        write(self.root/'prepared.json',receipt);seen=[]
        def add(path,expected):
            self.assertEqual(sha(path),expected);seen.append(path.name)
        binding=bind(self.root,graph,add);self.assertEqual(len(seen),3)
        self.assertEqual(binding['indptr']['array_sha256'],graph['csc_sha256']['indptr'])
        wrong=copy.deepcopy(graph);wrong['csc_sha256']['indptr']='0'*64
        with self.assertRaises(RuntimeError):descriptors(self.root,wrong)
        wrong=copy.deepcopy(graph);wrong['nodes']=3
        with self.assertRaises(RuntimeError):descriptors(self.root,wrong)
        prepared['indices']['sha256']=prepared['indices']['array_sha256'];receipt['bindings']['original_indices.npy']=prepared['indices']['sha256'];write(self.root/'prepared.json',receipt)
        with self.assertRaises(AssertionError):bind(self.root,graph,add)

    def test_live_csc_descriptor_schema_and_headers(self):
        from .graph_binding import descriptors
        p=read(HERE/'reference_protocol.json');data=ROOT/p['data'];d=descriptors(data,p['graph']);receipt=read(data/'prepared.json')
        for name,item in d.items():
            self.assertEqual(item['file_sha256'],receipt['bindings']['original_'+name+'.npy'])
            self.assertEqual(item['array_sha256'],p['graph']['csc_sha256'][name])
            self.assertNotEqual(item['file_sha256'],item['array_sha256'])

    def test_pool_bounds_identity_and_receipt_guards(self):
        from . import pool
        self.assertEqual(pool.spec('shared')['verify_bytes'],779018964992)
        self.assertGreater(pool.spec('shared')['capacity_bytes'],pool.spec('shared')['verify_bytes'])
        spec=dict(source=str(self.root/'source'),state=str(self.root/'state'),offset=4096,
                  capacity_bytes=8192,verify_bytes=8192,header_bytes=0,source_full_sha256='fixture')
        (self.root/'source').write_bytes(bytes(8192));spec['source_identity']=pool.identity(spec['source'])
        write(spec['state'],dict(status='verified',device='/dev/libnvm0',device_offset_bytes=4096,payload_bytes=8192,feature_file_sha256='fixture'))
        spec['state_sha256']=sha(spec['state'])
        # The receipt must cover the complete requested prefix and match both
        # the disk read and reference-source hash. It is not a mere file flag.
        with patch.object(pool,'POOL_PROOFS',self.root),patch.object(pool,'spec',return_value=spec):
            path=self.root/'pools/shared_receipt.json'
            r=dict(candidate_sha256=sha(pool.POOL_CANDIDATE),passed=True,mode='shared',raw_ssd_writes=False,pool_spec_sha256=sha(HERE/'pools.json'),
              state_sha256=spec['state_sha256'],verified_bytes=8192,offset=4096,all_finite=True,
              source_identity=spec['source_identity'],actual_sha256='a'*64,expected_sha256='a'*64)
            write(path,r);pool.check_receipt('shared')
            for key,value in [('verified_bytes',4096),('actual_sha256','b'*64),('all_finite',False),('state_sha256','bad')]:
                wrong=dict(r,**{key:value});write(path,wrong)
                with self.assertRaises(RuntimeError):pool.check_receipt('shared')

    def test_calibration_rejects_timing_or_io_and_sampling_changes(self):
        from .compare import collect
        modes=[('real','calibration/real_first'),('shared','native/g2_r20'),('shared','calibration/shared_repeat'),('real','calibration/real_second')]
        for mode,name in modes:
            f=self.root/name;f.mkdir(parents=True)
            report=dict(passed=True,feature_mode=mode,smoke=False,test=None,initial_parameters_sha256='i',initial_dgl_rng=[1],point={'id':'g2_r20'},
                layout_manifest_sha256='m',overlay_receipt_sha256='o',policy={},candidate_sha256='c',validation_seconds=0,
                io_accounting_validation=None,io_accounting_test=None,evaluation_lifecycle=dict(enabled=False,validation_calls=0,test_calls=0,diagnostic_replays=0,training_rng_unchanged=True,trace_preparation_seconds=0),
                io_accounting_training=dict(ssd_primary_bytes=1000,ssd_completed_bytes=1000,ssd_useful_bytes=500),
                epochs=[dict(train_seconds=100.,order_seconds=1.,updates=1,root_sha256='r',shapes=[1],windows=[],validation=None,
                             training=dict(feature=dict(cpu=10,gpu_ssd=20)))])
            write(f/'full_digit_full_accepted.json',report)
            write(f/'full_summary.json',dict(passed=True,report_sha256=dict(digit_full=sha(f/'full_digit_full_accepted.json'))))
            write(f/'status.json',dict(passed=True,complete=True,workers=[dict(mode=m,returncode=0) for m in ('smoke','full')]))
            audit={k:1 for k in ('inputs','outputs','blocks','storage_rows','storage_flags','cuda_rng','dgl_rng')}
            write(f/'smoke_digit_full_accepted.json',dict(audits=[audit]*4))
            write(f/'smoke_summary.json',dict(passed=True,report_sha256=dict(digit_full=sha(f/'smoke_digit_full_accepted.json'))))
        self.assertTrue(collect(self.root)['passed'])
        f=self.root/'native/g2_r20';original=read(f/'full_digit_full_accepted.json')
        for mutation in ('time','io','shape'):
            r=copy.deepcopy(original)
            if mutation=='time':r['epochs'][0]['train_seconds']=140.
            elif mutation=='io':r['io_accounting_training']['ssd_primary_bytes']=1200
            else:r['epochs'][0]['shapes']=[2]
            write(f/'full_digit_full_accepted.json',r)
            write(f/'full_summary.json',dict(passed=True,report_sha256=dict(digit_full=sha(f/'full_digit_full_accepted.json'))))
            with self.assertRaises(RuntimeError):collect(self.root)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU tests require hidden GPUs');setup()
    import torch
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(a.output,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
          cuda_initialized=torch.cuda.is_initialized(),raw_ssd_access=False,native_execution=False,points=getattr(Tests,'records',[])))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
