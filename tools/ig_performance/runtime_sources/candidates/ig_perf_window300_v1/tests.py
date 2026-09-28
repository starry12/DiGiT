"""Independent model math, 4KiB accounting and execution scope regressions."""
import copy,tempfile,unittest
from unittest import mock
import torch,numpy as np
from candidates.ig_perf_window300_v1.common import *
from candidates.ig_perf_window300_v1.model import make_model,optimizer
from candidates.pa_gat_native_v1.tests import blocks,oracle as gat_oracle
from candidates.pa_gcn_native_v1.tests import oracle as gcn_oracle
def sage_oracle(model,bs,x):
    h=x
    for i,(layer,b) in enumerate(zip(model.layers,bs)):
        u,v=b.edges(order='eid');dst=b.num_dst_nodes()
        means=torch.stack([h[u[v==node]].mean(0) for node in range(dst)])
        h=layer.fc_self(h[:dst])+layer.fc_neigh(means)
        if i<len(model.layers)-1:h=torch.relu(torch.nn.functional.dropout(h,p=model.dropout.p,training=model.training))
    return h
def math_check(name,device='cpu'):
    torch.manual_seed(71);model=make_model(name,device,torch.float64);ref=copy.deepcopy(model);bs=blocks(device)
    x=torch.randn(7,1024,device=device,dtype=torch.float64);labels=torch.tensor([0,18],device=device)
    opt=optimizer(model);other=optimizer(ref);errors=[]
    for step in range(3):
        torch.manual_seed(90+step);pred=model(bs,x);loss=torch.nn.functional.cross_entropy(pred,labels);opt.zero_grad(set_to_none=True);loss.backward()
        torch.manual_seed(90+step);want=dict(sage=sage_oracle,gcn=gcn_oracle,gat=gat_oracle)[name](ref,bs,x);expected=torch.nn.functional.cross_entropy(want,labels);other.zero_grad(set_to_none=True);expected.backward()
        torch.testing.assert_close(pred,want,rtol=1e-8,atol=1e-10);errors.append(float((pred-want).abs().max()))
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)
        opt.step();other.step()
        for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    return dict(passed=True,model=name,device=device,updates=3,max_abs_logits_error=max(errors),parameters=sum(p.numel() for p in model.parameters()))
class Checks(unittest.TestCase):
    def test_all_model_gradients_dropout_and_adam(self):
        for name in cfg()['models']:
            with self.subTest(model=name):self.assertTrue(math_check(name)['passed'])
    def test_npy_file_and_payload_hash_scopes_are_distinct(self):
        from candidates.ig_perf_window300_v1.inputs import payload_sha
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'values.npy';values=np.arange(32,dtype='int64');np.save(path,values);mapped=np.load(path,mmap_mode='r')
            self.assertEqual(payload_sha(path),sha(path));self.assertEqual(payload_sha(path,mapped.offset),digest(values));self.assertNotEqual(payload_sha(path),payload_sha(path,mapped.offset))
    def test_exact_schedule_has_disjoint_20_plus_3x100_roots(self):
        from candidates.ig_perf_window300_v1.windows import root_slices
        p=cfg();phases=root_slices(p);self.assertEqual(phases,[('warmup',0,20,20),('training',20,320,100)])
        seen=[]
        for _,lo,hi,width in phases:seen.extend(range(lo,hi))
        self.assertEqual(seen,list(range(320)))
        for key,value in [('validation',{}),('validation_calls',1),('test_calls',1),('epochs',1),('warmup_batches',19),('window_batches',29),('total_train_batches',111)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):root_slices(dict(p,**{key:value}))
    def test_missing_duplicate_or_nonfinite_window_rejected(self):
        from candidates.ig_perf_window300_v1.windows import check_windows
        seq=[dict(index=j,batch_start=j*30,batch_end=(j+1)*30,batches=30,seconds=float(j+1),group_edges=0,outer_edges=100) for j in range(3)]
        self.assertTrue(check_windows(seq,90,30,6.))
        for bad in [seq[:-1],seq+[seq[-1]],list(reversed(seq)),[dict(w,seconds=float('nan')) for w in seq],[dict(w,batch_start=0) for w in seq]]:
            with self.assertRaises(RuntimeError):check_windows(bad,90,30,6.)
        with self.assertRaises(RuntimeError):check_windows(seq,90,30,26.) # warmup must not leak into measured total
    def test_each_smoke_failure_blocks_full(self):
        from candidates.ig_perf_window300_v1 import controller as c
        for arm in ('gids','digit_full'):
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
                seen=[]
                def run(o,j,*args):
                    seen.append(j)
                    if j['arm']==arm:raise RuntimeError('fixture smoke failed')
                    return {}
                def bind(o,e):write(o/'inputs.json',{});return {}
                with mock.patch.object(c,'verify',return_value='candidate'),mock.patch('candidates.ig_perf_window300_v1.common.verify',return_value='entry'),mock.patch.object(c,'input_binding',side_effect=bind),mock.patch.object(c,'run_worker',side_effect=run):
                    with self.assertRaisesRegex(RuntimeError,'fixture smoke failed'):c.execute('representative',Path(td)/'fresh',2,'sage')
                self.assertTrue(all(j['mode']=='smoke' for j in seen))
    def test_v6_dispatch_and_dry_run(self):
        from submission.v9 import run
        for dataset in ('PA','IG'):
            for model in cfg()['models']:
                out=ROOT/'results/ig_unit_never_started'
                with mock.patch.object(run,'verify',return_value='entry'),mock.patch.object(run.subprocess,'call') as call,mock.patch('builtins.print') as log:
                    self.assertEqual(run.main(['representative','--dataset',dataset,'--model',model,'--gpu','2','--output',str(out),'--dry-run']),0)
                    v=json.loads(log.call_args[0][0]);self.assertIn('candidates.ig_perf_v5.controller' if dataset=='IG' else str(ROOT/'submission/v4/run.py'),v['command']);self.assertEqual(v['epochs'],None if dataset=='IG' else 20);call.assert_not_called();self.assertFalse(out.exists())
    def test_no_accepted_ig_or_old_full_alias(self):
        from submission.v9 import run
        for argv in (['full'],['summarize','--dataset','IG','--accepted','--output','results/unused']):
            with self.assertRaises(SystemExit),mock.patch('sys.stderr'):run.main(argv)
    def test_monitor_exit_terminates_worker_before_acceptance(self):
        from candidates.ig_perf_window300_v1 import controller as c
        class Child:
            pid=998
            def __init__(self):self.terminated=False
            def poll(self):return None
            def terminate(self):self.terminated=True
            def wait(self,timeout=None):return 0
        class Monitor:
            def __init__(self,*args):self.process=mock.Mock();self.process.poll.return_value=2;self.stopped=False
            def start(self):return dict(pid=997,passed=True)
            def stop(self):self.stopped=True;return 2
        child=Child();monitor=Monitor()
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
            out=Path(td);state=dict(pid=996,workers=[]);job=dict(mode='smoke',arm='gids',model='sage')
            with mock.patch.object(c,'check_device'),mock.patch.object(c,'ExternalMonitor',return_value=monitor),mock.patch.object(c.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'monitor exited during worker'):
                    c.run_worker(out,job,state,{},lambda **kw:None,2)
            self.assertTrue(child.terminated);self.assertTrue(monitor.stopped)
            self.assertFalse((out/'smoke_gids_accepted.json').exists())
    def test_four_kib_accounting_and_zero_io(self):
        from candidates.io_accounting_v1.accounting import validate_region,summarize
        u=dict(schema='digit-useful-io-v1',enabled=True,reconciled=True,ssd_fill_bytes=4096,ssd_useful_bytes=4096,gpu_feature_bytes=8192,gpu_feature_rows=2)
        d=dict(enabled=True,reconciled=True,completed_bytes=4096,primary_bytes=4096,replay_bytes=0,submitted_commands=1,completed_commands=1,active_ns=1000000)
        r=dict(useful_io=u,device=d,feature=dict(cpu=1,gpu_ssd=2),feature_row_bytes=4096,feature_seconds=.1)
        self.assertTrue(validate_region(r));self.assertEqual(summarize([(r,1)])['logical_feature_bytes'],12288)
        bad=copy.deepcopy(r);bad['feature_row_bytes']=512
        with self.assertRaises(RuntimeError):validate_region(bad)
        for k in ('ssd_fill_bytes','ssd_useful_bytes','gpu_feature_bytes','gpu_feature_rows'):u[k]=0
        for k in ('completed_bytes','primary_bytes','replay_bytes','submitted_commands','completed_commands','active_ns'):d[k]=0
        r['feature']['gpu_ssd']=0;self.assertIsNone(summarize([(r,1)])['ssd_useful_gbps'])
    def test_training_only_report_acceptance_and_checkpoint_step_rejection(self):
        from candidates.ig_perf_window300_v1.validation import report_check
        from candidates.ig_perf_window300_v1.model import model_config
        from candidates.io_accounting_v1.accounting import summarize
        p=cfg();model=make_model('sage');final=state_hash(model.state_dict())
        n=p['smoke_train_batches']*p['batch_size']
        window=dict(index=0,batch_start=0,batch_end=2,batches=2,seconds=1.,roots_sha256='roots',group_edges=0,outer_edges=1)
        v=dict(name='training',examples=n,batches=2,seconds=1.,phase_wall_seconds=1.1,roots_sha256='roots',losses=[1.,1.],sampling_shape_totals=[[3,2,1],[2,1,1],[1,n,1]],feature_rows=3,group_edges=0,outer_edges=1,window_count=1,windows=[window],feature_row_bytes=4096,feature_seconds=.1,
               feature=dict(cpu=1,gpu_ssd=2),useful_io=dict(schema='digit-useful-io-v1',enabled=True,reconciled=True,region_id=2,ssd_fill_bytes=4096,ssd_useful_bytes=4096,gpu_feature_bytes=8192,gpu_feature_rows=2),device=dict(enabled=True,reconciled=True,completed_bytes=4096,primary_bytes=4096,replay_bytes=0,submitted_commands=1,completed_commands=1,active_ns=1000000))
        r=dict(schema='digit-ig-window-report-v1',passed=True,arm='gids',smoke=True,candidate_sha256='candidate',protocol_sha256='protocol',model_name='sage',seed=0,epochs=None,model_config=model_config('sage'),optimizer=p['optimizer'],model_parameter_count=sum(x.numel() for x in model.parameters()),validation=None,validation_calls=0,test=None,test_calls=0,accuracy=None,final_accuracy_claim=False,epoch_time_claim=False,steady_state_proven=False,diagnostic_replays=0,raw_ssd_writes=False,all_metadata_and_cache_reused=True,route_warmup=True,updates=2,measured_batches=0,warmup=None,training=v,training_seconds=1.,io_accounting_training=summarize([(v,1.)]),initial_parameters_sha256='initial',final_parameters_sha256=final,native=dict(route_warmup=True,cache_rows=p['cpu_cache_rows'],native_info=dict(gpu_cache_bytes=p['gpu_cache_bytes'])),routing=dict(passed=True),source_audits=[dict(source_equal=True)]*2,admission=dict(passed=True,required_bytes=1024,host_required_bytes=2048),external_monitor=dict(observed_peak_device_used_bytes=512),peak_host_rss_kib=1)
        binding=dict(candidate_sha256='candidate',protocol_sha256='protocol',prepared_orders=dict(smoke_payload_sha256='roots'))
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);write(folder/'training.json',v)
            checkpoint=dict(epoch=None,updates=2,model_name='sage',model=model.state_dict(),optimizer=dict(state={0:dict(step=torch.tensor(2.))}))
            torch.save(checkpoint,folder/'final_model.pt');r['checkpoint_sha256']=sha(folder/'final_model.pt')
            self.assertTrue(report_check(r,'gids',True,binding,folder))
            for key,value in [('validation_calls',1),('test_calls',1),('measured_batches',90),('epochs',1),('updates',1),('accuracy',.5),('epoch_time_claim',True)]:
                with self.subTest(key=key),self.assertRaises(RuntimeError):report_check(dict(r,**{key:value}),'gids',True,binding,folder)
            checkpoint['optimizer']['state'][0]['step']=torch.tensor(1.)
            torch.save(checkpoint,folder/'final_model.pt');r['checkpoint_sha256']=sha(folder/'final_model.pt')
            with self.assertRaisesRegex(RuntimeError,'Optimizer update count differs'):report_check(r,'gids',True,binding,folder)
    def test_cross_window_companion_consumption_is_not_a_complete_region(self):
        from candidates.ig_perf_window300_v1.summarize import window_metrics
        from candidates.ig_perf_window300_v1.features import aggregate
        def window(fill,useful):
            return dict(device_raw=[0,0,fill//4096,fill//4096,fill,1000000],device_bytes=fill,primary_bytes=fill,replay_bytes=0,useful=dict(region_id=2,ssd_fill_bytes=fill,ssd_useful_bytes=useful,gpu_feature_bytes=8192,gpu_feature_rows=2),feature_cpu=0,feature_gpu_ssd=2,feature_seconds=.1)
        first=window(8192,4096);later=window(4096,8192)
        result=window_metrics(later);self.assertEqual(result['ssd_useful_consumed_bytes'],8192);self.assertIsNone(result['ssd_useful_gbps']);self.assertGreater(result['ssd_physical_gbps'],0)
        combined=aggregate([first,later]);self.assertEqual(combined['useful_io']['ssd_fill_bytes'],combined['useful_io']['ssd_useful_bytes'])
    def test_footer_exports_three_windows_without_accuracy_or_epoch_claim(self):
        from candidates.ig_perf_window300_v1.summarize import emit
        metrics=dict(ssd_completed_bytes=4096,ssd_useful_gbps=.5,ssd_physical_gbps=1.,effective_feature_gbps=2.)
        windows=[dict(index=j,batches=30,seconds=float(j+1),seconds_per_batch=(j+1)/30,training_roots_per_second=30720/(j+1),roots_sha256='fixture',group_edge_share=0.,io=metrics) for j in range(3)]
        row=dict(dataset='IG',model='sage',arm='gids',updates=110,warmup_batches=20,measured_batches=90,measurement_windows=3,training_seconds=6.,seconds_per_batch=6/90,training_roots_per_second=92160/6,warmup_seconds=2.,setup_seconds=2.,worker_seconds=10.,window_relative_spread=1.,outer_layer_group_edge_share=0.,cpu_feature_request_share=.1,observed_peak_gpu_bytes=1024,windows=windows,io=dict(training=metrics,warmup=metrics),monitor=dict(timeout_count=0))
        value=dict(rows=[row,dict(row,arm='digit_full')],strict_resource_acceptance=True,ratios=dict(training_speedup=1.),limitations=['fixture'],final_accuracy_claim=False,epoch_time_claim=False)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'summary';emit(value,out);self.assertFalse(read(out/'summary.json')['final_accuracy_claim']);self.assertFalse(read(out/'summary.json')['epoch_time_claim']);self.assertIn('No validation, test, accuracy or full-epoch claim',(out/'README.md').read_text());self.assertEqual(len((out/'windows.csv').read_text().splitlines()),7)
if __name__=='__main__':
    setup();torch.set_num_threads(1);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),indent=2));sys.exit(not result.wasSuccessful())
