"""Small, isolated correctness and guard checks; no real graph scan/framework/GPU."""
import copy,fcntl,hashlib,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import ROOT,cfg,sha,data_root,storage_snapshot,receipt_path
from .inventory import header,audit,capacity
from .safety import Busy,exclusive_idle,ensure_ig_closed
from .synthetic import generate
from .benchmark_core import measure
from .model import model_config

class Fixture(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(prefix='digit_uks_fixture_');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def source(self):
        p=copy.deepcopy(cfg());p.update(nodes=32,source_edges=6)
        np.save(self.root/'edge_index.npy',np.array([[1,2,2,4,8,31],[0,2,2,1,0,31]],dtype='<i8'))
        labels=np.arange(32,dtype='<f4').reshape(-1,1);labels[0]=np.nan;np.save(self.root/'node_label.npy',labels)
        (self.root/(p['upstream']+'.properties')).write_text('nodes=32\narcs=6\n')
        return p
    def small(self):
        p=copy.deepcopy(cfg());p.update(nodes=128,feature_dim=4,batch_size=1);p['training']['train_nodes']=112;p['synthetic']['block_rows']=16
        return p
    def test_source_samples_and_nonfinite_old_labels(self):
        p=self.source();v=audit(self.root,p);self.assertFalse(v['full_source_validated']);self.assertFalse(v['bulk_scan']);self.assertGreater(v['legacy_labels']['nonfinite_samples'],0);self.assertLessEqual(v['raw_payload_bytes_read'],480)
    def test_source_shape_rejects_transposed(self):
        p=self.source();np.save(self.root/'edge_index.npy',np.zeros((6,2),dtype='<i8'))
        with self.assertRaises(ValueError):audit(self.root,p)
    def test_source_truncation_rejected(self):
        self.source();p=self.root/'edge_index.npy';p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):header(p)
    def test_sampled_bad_endpoint_rejected(self):
        p=self.source();a=np.load(self.root/'edge_index.npy');a[0,0]=32;np.save(self.root/'edge_index.npy',a)
        with self.assertRaises(ValueError):audit(self.root,p)
    def test_source_property_mismatch(self):
        p=self.source();(self.root/(p['upstream']+'.properties')).write_text('nodes=31\narcs=6\n')
        with self.assertRaises(ValueError):audit(self.root,p)
    def test_capacity_is_not_admission(self):
        d=capacity();self.assertFalse(d['gpu_admission_passed']);self.assertFalse(d['raw_ssd_payload_ready']);self.assertIsNone(d['ssd_offsets']);self.assertEqual(d['original_csc_runtime_upper_bytes'],8*(133633040+1+2*(5507679822+133633040)))
    def test_busy_lock_is_refused(self):
        p=self.root/'lock';p.touch()
        with p.open('rb') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(Busy):
                with exclusive_idle((p,)):pass
    def test_missing_lock_is_unknown_not_idle(self):
        with self.assertRaises(Busy):
            with exclusive_idle((self.root/'missing',)):pass
    def test_lock_symlink_rejected(self):
        p=self.root/'real';p.touch();link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(Busy):
            with exclusive_idle((link,)):pass
    def test_all_locks_released_on_partial_failure(self):
        a,b=self.root/'a',self.root/'b';a.touch();b.touch()
        with b.open('rb') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(Busy):
                with exclusive_idle((a,b)):pass
            with exclusive_idle((a,)):pass
    def test_ig_live_rejected(self):
        value=dict(known=True,stage='full_gids',processes=[dict(live_matching_ig=True)])
        with patch('candidates.uks_sage_v2.safety.active_ig',return_value=value):
            with self.assertRaises(Busy):ensure_ig_closed()
    def test_ig_unknown_rejected(self):
        with patch('candidates.uks_sage_v2.safety.active_ig',return_value=dict(known=False)):
            with self.assertRaises(Busy):ensure_ig_closed()
    def test_ig_pid_reuse_and_completed(self):
        value=dict(known=True,stage='complete',processes=[dict(live_matching_ig=False)])
        with patch('candidates.uks_sage_v2.safety.active_ig',return_value=value):self.assertEqual(ensure_ig_closed(),value)
    def test_synthetic_determinism_and_split(self):
        p=self.small();a=generate(self.root/'a',p);b=generate(self.root/'b',p)
        for name in a['files']:self.assertEqual(a['files'][name]['sha256'],b['files'][name]['sha256']);self.assertEqual(a['files'][name]['sha256'],sha(self.root/'a'/name))
        f=np.load(self.root/'a/node_feat.npy');y=np.load(self.root/'a/node_label.npy');roots=np.load(self.root/'a/benchmark_roots.npy');train=np.load(self.root/'a/train_nodes.npy')
        self.assertTrue(np.isfinite(f).all());self.assertTrue((f>=-1).all() and (f<1).all());self.assertTrue((y>=0).all() and (y<19).all());self.assertEqual(len(np.unique(roots)),110);self.assertTrue(np.isin(roots,train).all());self.assertFalse(a['native_ready']);self.assertFalse(a['accuracy_claim'])
    def test_feature_seed_does_not_change_labels(self):
        p=self.small();a=generate(self.root/'a',p);p['synthetic']['feature_seed']+=1;b=generate(self.root/'b',p)
        self.assertNotEqual(a['files']['node_feat.npy']['sha256'],b['files']['node_feat.npy']['sha256']);self.assertEqual(a['files']['node_label.npy']['sha256'],b['files']['node_label.npy']['sha256'])
    def test_insufficient_roots_rejected_before_write(self):
        p=self.small();p['training']['train_nodes']=10
        with self.assertRaises(ValueError):generate(self.root/'bad',p)
        self.assertFalse((self.root/'bad').exists())
    def test_synthetic_never_overwrites(self):
        p=self.small();generate(self.root/'a',p)
        with self.assertRaises(FileExistsError):generate(self.root/'a',p)
    def test_normalized_csc_stable_multiedges(self):
        # Includes repeated nonself edges, old self-loops, and isolated nodes.
        edges=np.array([[2,1,2,2,0,3],[1,1,1,2,1,0]],dtype='<i8');features=np.zeros((5,4),dtype='<f4')
        ep,fp=self.root/'edges.npy',self.root/'features.npy';np.save(ep,edges);np.save(fp,features)
        cmd=[sys.executable,str(ROOT/'evaluation/digit/normalized_csc.py'),'--edges',str(ep),'--features',str(fp),'--workspace',str(self.root/'csc'),'--chunk-edges','2','--fan-in','2','--memory-mib','64','--address-space-mib','512']
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=30,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'));self.assertEqual(r.returncode,0,r.stderr)
        ptr=np.load(self.root/'csc/original_indptr.npy');idx=np.load(self.root/'csc/original_indices.npy');contract=json.loads((self.root/'csc/graph_source.json').read_text())
        expected=[(int(a),int(b)) for a,b in edges.T if a!=b]+[(i,i) for i in range(5)]
        for dst in range(5):self.assertEqual(idx[ptr[dst]:ptr[dst+1]].tolist(),[a for a,b in expected if b==dst])
        self.assertEqual(contract['removed_self_edges'],2);self.assertEqual(contract['num_edges'],9)
    def test_sage_config_shape(self):
        v=model_config();self.assertEqual(v['in_feats'],256);self.assertEqual(v['num_classes'],19);self.assertEqual(v['num_layers'],3)
    def timed(self,variable=False):
        p=self.small();clock=[0.0];seen=[];begins=[]
        def step(rows):
            i=len(seen);seen.extend(rows);clock[0]+=(.2 if variable and i>=80 else .1)
        v=measure(list(range(110)),step,lambda:None,lambda:dict(updates=len(seen)),lambda:begins.append(len(seen)),p,clock=lambda:clock[0])
        return v,seen,begins
    def test_window_excludes_warmup_and_keeps_all_steps(self):
        v,seen,begins=self.timed();self.assertEqual(seen,list(range(110)));self.assertEqual(begins,[20]);self.assertAlmostEqual(v['warmup_seconds'],2);self.assertAlmostEqual(v['measured_seconds'],9);self.assertEqual(v['measured_batches'],90);self.assertIsNone(v['epoch_time_seconds']);self.assertFalse(v['native_acceptance']);self.assertFalse(v['steady_state_proven'])
    def test_window_instability_not_discarded(self):
        v,_,_=self.timed(True);self.assertFalse(v['window_stability_heuristic_passed']);self.assertEqual(len(v['windows']),3);self.assertAlmostEqual(v['measured_seconds'],12)
    def test_timing_root_count_rejected(self):
        with self.assertRaises(ValueError):measure([1],lambda r:None,lambda:None,lambda:{},lambda:None,self.small())
    def test_external_data_root_no_creation(self):
        path='/mnt/n0/digit_ae_data/uks_sage_fixture_plan_only'
        with patch.dict(os.environ,{'DIGIT_UKS_DATA_ROOT':path}):
            self.assertEqual(str(data_root()),path);self.assertFalse(Path(path).exists());snap=storage_snapshot();self.assertEqual(snap['data_root'],path);self.assertFalse(snap['directory_created'])
    def test_alternate_n3_data_root(self):
        with patch.dict(os.environ,{'DIGIT_UKS_DATA_ROOT':'/mnt/n3/digit_ae_data/uks_sage_fixture_plan_only'}):self.assertEqual(str(data_root()),'/mnt/n3/digit_ae_data/uks_sage_fixture_plan_only')
    def test_relative_data_root_rejected(self):
        with patch.dict(os.environ,{'DIGIT_UKS_DATA_ROOT':'data/relative'}):
            with self.assertRaises(ValueError):data_root()
    def test_data_root_cannot_replace_source(self):
        with patch.dict(os.environ,{'DIGIT_UKS_DATA_ROOT':'/mnt/n3/ukunion','DIGIT_UKS_SOURCE':'/mnt/n3/ukunion'}):
            with self.assertRaises(ValueError):data_root()
    def test_external_receipt_path(self):self.assertEqual(receipt_path('/mnt/n0/digit_ae_data/test/ready.json'),'/mnt/n0/digit_ae_data/test/ready.json')
    def test_no_accidental_framework_import(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('dgl',sys.modules)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Fixture);result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),raw_ssd_access=False,torch_imported='torch' in sys.modules,gpu_checks=False)))
    sys.exit(0 if result.wasSuccessful() else 1)
