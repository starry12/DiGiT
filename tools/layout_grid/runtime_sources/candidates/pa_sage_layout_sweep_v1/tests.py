"""CPU-only small graphs: every grid cell, features, budgets and reverse edges."""
import argparse
import copy
import resource
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .common import *
from .build import Context,build_layout,exclusive
from .cpu_runtime import module,artifacts
from .metadata import group,validate_metadata
from .overlay import build_overlay

def fixture(root,n=96,hot_count=8,empty=False):
    root.mkdir()
    edges=[] if empty else [(int((v+k)%n),v) for v in range(n) for k in range(1,20)]+[(0,1),(0,1)]
    e=len(edges)
    def csc(records):
        records=sorted(records,key=lambda x:x[1])
        counts=np.bincount([r[1] for r in records],minlength=n)
        return np.r_[0,np.cumsum(counts)].astype(np.int64),np.array([r[0] for r in records],dtype=np.int64),np.array([r[2] for r in records],dtype=np.int64)
    original=[(u,v,i) for i,(u,v) in enumerate(edges)]
    # Empty fixtures exercise zero-length files, not the PA graph policy.
    loops=[] if empty else [(v,v,e+v) for v in range(n)]
    ptr,idx,_=csc(original+loops)
    bp,bi,be=csc(original+[(v,u,e+i) for i,(u,v) in enumerate(edges)]+([] if empty else [(v,v,2*e+v) for v in range(n)]))
    features=np.random.default_rng(7).normal(size=(n,128)).astype('float32')
    values=dict(indptr=ptr,indices=idx,features=features,hot_nodes=np.arange(hot_count,dtype=np.int64))
    inputs={}
    for name,value in values.items():
        path=root/(name+'.npy');save(path,value);inputs[name]=dict(path=str(path),sha256=sha(path))
    reverse=root/'bidirectional';reverse.mkdir()
    for name,value in zip(('indptr','indices','eids'),(bp,bi,be)):save(reverse/('original_'+name+'.npy'),value)
    return inputs,len(idx),reverse,e

class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='digit-layout-cpu-')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_all_fifteen_cells_and_overlays(self):
        inputs,edges,bidir,nonself=fixture(self.root/'inputs')
        source_before={d['path']:sha(d['path']) for d in inputs.values()}
        self.assertEqual(len({p['id'] for p in grid()}),15)
        achieved={}
        for spec in grid():
            with self.subTest(point=spec['id']):
                ctx=Context(spec,inputs,96,edges,8,self.root/spec['id'],fixture=True)
                result=build_layout(ctx);m=read(ctx.output/'final/bundle/manifest.json')
                self.assertEqual(m['grouping']['replication_ratio'],spec['replication_ratio'])
                self.assertEqual(m['grouping']['group_size'],spec['group_size'])
                self.assertLessEqual(result['replica_rows_used'],96*spec['replica_percent']//100)
                self.assertEqual(result['replica_rows_used']%spec['group_size'],0)
                b=budget(96,edges,8,spec['group_size'],spec['replica_percent'])
                self.assertLessEqual(result['storage_rows'],b['storage_rows_max'])
                rows=array(ctx.output/'full_cpu_rows.npy');mapping=array(ctx.output/'final/bundle/storage_to_node.npy')
                self.assertTrue(np.array_equal(mapping[rows],np.arange(8)))
                self.assertTrue(np.array_equal(rows.reshape(-1,8),rows[::8,None]+np.arange(8)))
                self.assertTrue(np.all(rows[::8]%8==0))
                if spec['replica_percent']==0:
                    self.assertEqual(m['grouping']['num_replica_groups'],0)
                    self.assertGreater(m['grouping']['num_primary_groups'],0)
                overlay=build_overlay(ctx.output,inputs['indptr']['path'],bidir,nonself,ctx.output/'overlay',fixture=True)
                self.assertTrue(overlay['passed']);self.assertFalse(overlay['native_ready'])
                achieved[spec['id']]=result
        for path,digest in source_before.items():self.assertEqual(sha(path),digest)
        for g in GROUPS:
            counts=[achieved[point(g,r)['id']]['replica_rows_used'] for r in PERCENTS]
            self.assertEqual(counts,sorted(counts))

    def test_grouping_matches_independent_reference_without_hot(self):
        inputs,e,_,_=fixture(self.root/'inputs',hot_count=0)
        ptr=array(inputs['indptr']['path']);idx=array(inputs['indices']['path']);order=np.argsort(-np.diff(ptr),kind='stable')
        for spec in grid():
            chunks=[];owners=[]
            def sink(phase,owner,members,offset):chunks.append(members.copy());owners.extend([owner]*len(members))
            result=group(ptr,idx,order,np.zeros(96,dtype=bool),np.array([],dtype=np.int64),g=spec['group_size'],replica_percent=spec['replica_percent'],sink=sink)
            reference=module('reorganization').build_groups(ptr,idx,96,group_size=spec['group_size'],replication_ratio=spec['replication_ratio'],seed=0)
            self.assertTrue(np.array_equal(np.concatenate(chunks),reference.group_members))
            self.assertEqual(owners,reference.group_owner.tolist())
            self.assertEqual(result['primary'],reference.num_primary_groups)

    def test_empty_groups_and_hot_only(self):
        for empty,hot in ((True,0),(False,16)):
            name='empty' if empty else 'all_hot'
            inputs,edges,_,_=fixture(self.root/(name+'_inputs'),n=16,hot_count=hot,empty=empty)
            ctx=Context(point(4,80),inputs,16,edges,hot,self.root/name,fixture=True)
            r=build_layout(ctx)
            self.assertEqual(r['replica_rows_used'],0)
            self.assertEqual(read(ctx.output/'final/bundle/manifest.json')['grouping']['num_groups'],0)

    def test_hash_corruption_and_overwrite_rejected(self):
        inputs,edges,bidir,nonself=fixture(self.root/'inputs')
        ctx=Context(point(2,20),inputs,96,edges,8,self.root/'cell',fixture=True)
        build_layout(ctx)
        with self.assertRaisesRegex(RuntimeError,'already exists'):build_layout(ctx)
        wrong_ptr=np.load(inputs['indptr']['path']);wrong_ptr[1]-=1
        save(self.root/'wrong_ptr.npy',wrong_ptr)
        with self.assertRaisesRegex(RuntimeError,'Directed graph differs'):
            build_overlay(ctx.output,self.root/'wrong_ptr.npy',bidir,nonself,self.root/'bad_overlay',fixture=True)
        self.assertFalse((self.root/'bad_overlay').exists())
        path=ctx.output/'final/bundle/node_to_primary_row.npy'
        a=np.load(path);a[0]=a[1];np.save(path,a)
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):validate_metadata(ctx,ctx.output,ctx.output)
        bad=copy.deepcopy(inputs);bad['features']['sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'hash/identity mismatch'):
            build_layout(Context(point(1,0),bad,96,edges,8,self.root/'bad',fixture=True))

    def test_invalid_parameters_and_fixture_limits(self):
        for g,r in ((True,20),(3,20),(2,25),(2,-1),(2,20.0)):
            with self.assertRaises(RuntimeError):point(g,r)
        inputs,edges,_,_=fixture(self.root/'inputs')
        with self.assertRaisesRegex(RuntimeError,'lightweight limits'):
            build_layout(Context(point(2,20),inputs,4097,edges,8,self.root/'large',fixture=True))
        self.assertFalse((self.root/'large').exists())

    def test_lock_excludes_concurrent_builds(self):
        lock=self.root/'lock';lock.touch()
        with exclusive(lock):
            with self.assertRaises(BlockingIOError):
                with exclusive(lock):pass
        with exclusive(lock):pass

    def test_production_entry_takes_lock_before_inputs(self):
        ctx=Context(point(2,20),{},111059956,1726745828,11105992,self.root/'nested'/'cell')
        with patch('candidates.pa_sage_layout_sweep_v1.build.exclusive',side_effect=BlockingIOError('active ablation')):
            with self.assertRaises(BlockingIOError):build_layout(ctx)
        self.assertFalse(ctx.output.exists())
        self.assertEqual(existing_parent(ctx.output.parent),self.root)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(),'Test output must be fresh')
    before=time.time()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SweepTests))
    require(not any(name in sys.modules for name in ('torch','dgl','BAM_Feature_Store')),'CPU tests imported a GPU runtime')
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                points=15,seconds=time.time()-before,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                gpu_runtime_imported=False,raw_ssd_access=False,large_inputs_loaded=False,
                scope='synthetic CPU layout, feature, reverse-edge, boundary and exclusion tests; no PA runtime acceptance')
    write(a.output,record);print(json.dumps(record,indent=2))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
