"""Process-local wrappers: unchanged sampling calls, exclusive nested wall time.

CUDA stage boundaries synchronize completion. These are diagnostic synchronized
wall durations, not pure kernel timings or a replacement throughput benchmark.
"""
import time
from contextlib import contextmanager

KEYS=('batch_other','sampling_other','frontier_other','native_sample_other',
      'cpu_owner_validation','native_buffers_and_transfer','native_cuda_entry',
      'cpu_output_validation','cpu_group_validation','cpu_frontier','cpu_block',
      'gpu_numbering','gpu_block','gpu_storage_rows','request_mapping',
      'feature_fetch','block_label_transfer','forward_loss','zero_grad',
      'backward','runtime_check','adam')


class Timer:
    def __init__(self,synchronize,clock=time.perf_counter,cpu_clock=time.thread_time):
        self.sync=synchronize;self.clock=clock;self.cpu_clock=cpu_clock
        self.active=False;self.stack=[];self.batches=0
        self.stats={k:dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.,
                           thread_cpu_exclusive_seconds=0.,boundary_sync_seconds=0.) for k in KEYS}

    @contextmanager
    def measure(self,name,cuda=False):
        if name not in self.stats:raise ValueError('Unknown profiling stage')
        if not self.active:
            try:yield
            finally:
                if cuda:self.sync()
            return
        frame=[self.clock(),self.cpu_clock(),0.,0.];self.stack.append(frame);wait=0.
        try:yield
        finally:
            try:
                if cuda:
                    start=self.clock();self.sync();wait=self.clock()-start
            finally:
                elapsed=self.clock()-frame[0];cpu=self.cpu_clock()-frame[1]
                if self.stack.pop() is not frame:raise RuntimeError('Profile nesting changed')
                v=self.stats[name];v['calls']+=1;v['inclusive_seconds']+=elapsed
                v['exclusive_seconds']+=max(0.,elapsed-frame[2])
                v['thread_cpu_exclusive_seconds']+=max(0.,cpu-frame[3])
                v['boundary_sync_seconds']+=wait
                if self.stack:self.stack[-1][2]+=elapsed;self.stack[-1][3]+=cpu

    def report(self):
        if self.stack:raise RuntimeError('Profile has unfinished intervals')
        total=self.stats['batch_other']['inclusive_seconds']
        exclusive=sum(v['exclusive_seconds'] for v in self.stats.values())
        if abs(total-exclusive)>max(1e-6,total*1e-9):raise RuntimeError('Profile double-counted intervals')
        return dict(schema='cl-symmetric-stages-v1',measured_batches=self.batches,
                    synchronized_wall=True,pure_kernel_timing=False,exclusive_sum_seconds=exclusive,
                    batch_wall_seconds=total,stages=self.stats,
                    boundary_sync_seconds=sum(v['boundary_sync_seconds'] for v in self.stats.values()))


class Hooks:
    def __init__(self,timer,native,sampler,arm):
        self.timer,self.native,self.sampler,self.arm=timer,native,sampler,arm
        self.originals=[]

    def wrap(self,obj,name,stage,cuda=False):
        original=getattr(obj,name);own=name in vars(obj);timer=self.timer
        def call(*args,**kwargs):
            with timer.measure(stage,cuda):return original(*args,**kwargs)
        self.originals.append((obj,name,own,original));setattr(obj,name,call)

    def __enter__(self):
        try:
            for method,stage in [('sample','native_sample_other'),('_owners','cpu_owner_validation'),
                                 ('_cuda','native_buffers_and_transfer'),('fn','native_cuda_entry'),
                                 ('_validate_output','cpu_output_validation'),('_validate_group_rows','cpu_group_validation')]:
                # sample_cuda already calls cudaDeviceSynchronize; preserve it.
                self.wrap(self.native,method,stage)
            self.wrap(self.sampler,'layers','frontier_other',cuda=True)
            self.wrap(self.sampler,'sample_blocks','sampling_other',cuda=True)
            if self.arm=='digit':
                from . import device
                for name,stage in [('number_layer','gpu_numbering'),('make_block','gpu_block'),('storage_rows','gpu_storage_rows')]:
                    self.wrap(device,name,stage,cuda=True)
            elif self.arm=='gids':
                from candidates.ukl_native_sampling_v10 import sampling
                from candidates.ukl_sage_compact_v1 import sampler
                self.wrap(sampling,'stable_unique','cpu_frontier')
                self.wrap(sampler,'block_from_edges','cpu_block')
            else:raise ValueError('Unknown profiling arm')
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        for obj,name,own,original in reversed(self.originals):
            if own:setattr(obj,name,original)
            else:delattr(obj,name)
        self.originals.clear()
