"""CUDA checkpoints only: no background sampler and no worker subprocesses."""
import os
from pathlib import Path
import time
from ae.pa_sage.common import write


class CheckpointObservations:
    def __init__(self,output):
        self.output=Path(output);self.checkpoints=[]
    def mark(self,stage):
        import torch
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        self.checkpoints.append(dict(stage=stage,time_unix=time.time(),device_free_bytes=free,
            device_used_bytes=total-free,device_total_bytes=total,
            torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(self.output,self.result())
    def result(self):
        return dict(mode='checkpoints_only_external_sampler',worker_pid=os.getpid(),checkpoints=self.checkpoints,
            observed_peak_device_used_bytes=max([0]+[x['device_used_bytes'] for x in self.checkpoints]),
            monitor_samples=0,monitor_error=None,background_monitor_in_worker=False,
            limitation='Worker stores CUDA checkpoints only; continuous GPU samples are collected by the controller-owned external process.')
    def __enter__(self):
        import torch
        torch.cuda.reset_peak_memory_stats();self.mark('start');return self
    def __exit__(self,*exc):write(self.output,self.result())
