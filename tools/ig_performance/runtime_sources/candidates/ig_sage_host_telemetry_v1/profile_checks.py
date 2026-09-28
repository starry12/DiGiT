"""CPU-only tests of the actual transformed loop, source parity and fail-closed gates."""
import ast
import copy
import ctypes
import hashlib
import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from .common import HERE, ROOT, OUT, cfg, require, setup, sha, write
from .profile import HostProfile, TOP, dump, transform, StripSpans, instrument_function, install, validate

setup()
import dgl
from .model import make_model, optimizer
from . import worker
from .features import NativeFeatures


def tree_function(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)


class RemoveHooks(ast.NodeTransformer):
    def visit_Expr(self,node):
        c=node.value
        if isinstance(c,ast.Call):
            if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='PROFILE':
                return None
            if isinstance(c.func,ast.Name) and c.func.id=='install':
                return None
            if isinstance(c.func,ast.Name) and c.func.id=='write' and "stage_profile.json" in dump(c):
                return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if dump(node.targets[0])==dump(ast.parse("report['stage_profile']",mode='eval').body).replace('Load()', 'Store()', 1):
            return None
        if isinstance(node.targets[0],ast.Subscript) and "stage_profile" in dump(node.targets[0]):
            return None
        return self.generic_visit(node)


class FakeStore:
    def __init__(self, features, values):self.features,self.values=features,values
    def begin_useful_io_region(self):pass
    def get_useful_io_stats(self):return [0,2]
    def read_feature(self,out,ids,count,dim,cache_dim,offset,flags):
        index=np.ctypeslib.as_array((ctypes.c_int64*count).from_address(ids))
        dest=np.ctypeslib.as_array((ctypes.c_float*(count*dim)).from_address(out)).reshape(count,dim)
        dest[:]=self.values[index]


class FakeFeatures(NativeFeatures):
    """Real fetch method and real tensors, but host memory instead of SSD/CUDA."""
    def __init__(self, values):
        self.arm='gids';self.plan=dict(rows=len(values));self.feature_seconds=0.
        self.store=FakeStore(self,values)
    def begin(self):return self.feature_seconds
    def finish(self,before,count):
        return dict(rows=count,feature_seconds=self.feature_seconds-before,feature_cpu=count,feature_gpu_ssd=0)


def run_cpu_loop(mode, folder):
    # Execute the actual nested fetch/phase functions from worker.execute.
    profile=HostProfile(mode)
    execute=tree_function(HERE/'worker.py','execute')
    module=ast.Module(body=[execute],type_ignores=[])
    if mode=='host':module,_=transform(module,'worker')
    functions=[n for n in module.body[0].body if isinstance(n,ast.FunctionDef) and n.name in ('fetch','phase')]
    torch.manual_seed(72);dgl.seed(72)
    graph=dgl.graph((torch.arange(24).repeat_interleave(5), (torch.arange(24).repeat_interleave(5)+torch.arange(5).repeat(24))%24),num_nodes=24)
    sampler=dgl.dataloading.NeighborSampler([10,5,5],replace=False)
    model=make_model('sage');opt=optimizer(model)
    values=np.random.RandomState(42).normal(size=(24,1024)).astype('float32')
    features=FakeFeatures(values);install(profile,sampler,graph,features)
    labels=np.arange(24,dtype='float32')%19
    config=dict(cfg(),batch_size=2)
    def aggregate(windows):
        return dict(useful_io=dict(region_id=2),feature=dict(cpu=sum(w['rows'] for w in windows),gpu_ssd=0))
    trace=SimpleNamespace(window=mock.Mock())
    ns=dict(trace=trace,np=np,torch=torch,dgl=dgl,hashlib=hashlib,time=time,PROFILE=profile,
        retained=lambda:None,model=model,opt=opt,features=features,sampler=sampler,
        graph=graph,arrays=None,labels=labels,p=config,require=require,
        a=SimpleNamespace(output=folder,arm='gids',model='sage',smoke=False),
        aggregate=aggregate,append_sync=lambda *a,**k:None,progress=lambda *a,**k:None,
        write=write,plan=dict(required_bytes=1024),raw=None,audits=[])
    original_empty=torch.empty
    def empty(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_empty(*args,**kwargs)
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(HERE/'worker.py')+':cpu-fixture','exec'),ns)
        # This fixture replaces existing GPU APIs, never invokes a CUDA runtime.
        with mock.patch.object(torch.Tensor,'cuda',lambda tensor,*a,**k:tensor), \
             mock.patch.object(torch,'empty',side_effect=empty), \
             mock.patch.object(torch.cuda,'synchronize') as synchronize, \
             mock.patch.object(torch.cuda,'mem_get_info',return_value=(10**12,10**12)):
            result=ns['phase']('training',np.arange(8,dtype='int64'),sampler,2)
        return dict(phase=result,profile=profile.result(),model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(opt.state_dict()),synchronize_calls=synchronize.call_count,markers=trace.window.call_args_list)
    finally:profile.close()


class Checks(unittest.TestCase):
    def test_same_host_loop_loss_adam_and_sync_as_previous_night(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            with mock.patch(__name__+'.HERE',ROOT/'candidates/ig_sage_eid_warp_stop_v1'):
                old=run_cpu_loop('host',folder/'old')
            new=run_cpu_loop('host',folder/'new')
        self.assertEqual(old['phase']['losses'],new['phase']['losses'])
        self.assertEqual(old['phase']['sampling_shape_totals'],new['phase']['sampling_shape_totals'])
        self.assertEqual(old['synchronize_calls'],new['synchronize_calls'])
        for k,v in old['model'].items():self.assertTrue(torch.equal(v,new['model'][k]))
        for k,state in old['optimizer']['state'].items():
            for name,value in state.items():self.assertTrue(torch.equal(value,new['optimizer']['state'][k][name]))
        self.assertEqual(len(new['markers']),2)
        for marker,window in zip(new['markers'],new['phase']['windows']):
            self.assertEqual(marker.args[0],'training');self.assertEqual(marker.args[1],window['index']);self.assertEqual(marker.args[3],window['seconds'])
        for result in (old,new):self.assertTrue(validate(dict(stage_profile=result['profile'],training=result['phase'],warmup=None,arm='gids')))

    def test_instrumentation_still_strips_to_original_execution(self):
        for filename,name,kind in [('worker.py','execute','worker'),('features.py','fetch','features')]:
            original=ast.Module(body=[tree_function(HERE/filename,name)],type_ignores=[])
            instrumented,sites=transform(copy.deepcopy(original),kind)
            self.assertEqual(dump(StripSpans().visit(instrumented)),dump(original))
            self.assertTrue(all(n==1 for n in sites.values()))

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main(verbosity=2)
