"""Replay four fresh arms with accepted preparation and no continuous GPU monitor."""
import csv
import os
from pathlib import Path
import signal
import time
from contextlib import ExitStack
from .common import HERE,GRID,PY,MODULE,ARMS,read,sha,write,require,heavy_gate,lock,verify
from .protocol import validate,schedule
from .process import run_worker
from . import reuse
from candidates.pa_sage_cache_policy_v3.controller import Journal as PriorJournal,stage_key,wait_available
from .admission import estimate
from .common import binary_receipt


class Journal(PriorJournal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.save(schema='digit-cache-controller-v6',continuous_gpu_monitoring=False)


def ensure_short(journal,p,accepted,protocol_path,prepared,execution):
    from .validation import make_smoke_matrix,smoke_matrix
    value=make_smoke_matrix({a:accepted['smoke_'+a] for a in ARMS},p,execution,
                           sha(protocol_path),sha(prepared),binary_receipt()['binary_sha256'])
    path=journal.output/'short_matrix.json'
    if path.exists():require(read(path)==value,'Short matrix changed on resume')
    else:write(path,value)
    smoke_matrix(read(path),p,execution,sha(protocol_path),sha(prepared))
    return path


def run(protocol_path,output,resume=False):
    heavy_gate();p=validate(read(protocol_path));execution=verify();binary_receipt()
    with lock(GRID/'controller.lock',create=False):heavy_gate()
    with ExitStack() as stack:
        for name in ('/run/digit-ae-selfservice/exclusive.lock','/tmp/digit-pa-bidir-controller.lock',
                     '/tmp/digit-pa-cache-policy-v6-controller.lock'):
            stack.enter_context(lock(name))
        journal=Journal(output,execution,sha(protocol_path),resume);accepted={}
        def stop(signum,frame):raise KeyboardInterrupt('Cache controller signaled')
        old_term=signal.signal(signal.SIGTERM,stop);old_int=signal.signal(signal.SIGINT,stop)
        try:
            for item in schedule(p):
                heavy_gate();require(verify()==execution,'Candidate changed during run')
                stage,arm=item['stage'],item['arm'];key=stage_key(item)
                previous=journal.previous(key)
                if previous is not None:
                    accepted[key]=previous
                    if stage=='reuse_preparation':reuse.check(previous,p,protocol_path,execution)
                    continue
                attempt=journal.attempt(key)
                if stage=='reuse_preparation':
                    path=reuse.materialize(p,protocol_path,attempt,execution)
                    binding,prepared=reuse.check(path,p,protocol_path,execution)
                else:
                    binding,prepared=reuse.check(accepted['reuse_preparation'],p,protocol_path,execution)
                    if stage in ('smoke','full'):
                        short=ensure_short(journal,p,accepted,protocol_path,prepared,execution) if stage=='full' else None
                        wait_available(estimate(p,read(binding)['manifest'],arm),journal);journal.save(stage=key)
                        command=[PY,'-B','-u','-m',MODULE+'.worker','--mode',stage,'--arm',arm,
                            '--protocol',str(protocol_path),'--binding',str(binding),'--prepared',str(prepared),
                            '--output',str(attempt/'worker'),'--controller-pid',str(os.getpid())]
                        if short is not None:command+=['--short-receipt',str(short)]
                        report,row=run_worker(command,attempt,arm,lambda **kw:journal.save(worker=kw))
                        from .validation import short as accept_short,full as accept_full
                        args=(report,p,arm,execution,sha(protocol_path),sha(prepared))
                        if stage=='smoke':accept_short(*args,binary_receipt()['binary_sha256'])
                        else:accept_full(*args,short,binary_receipt()['binary_sha256'])
                        path=attempt/'accepted.json';write(path,report)
                    else:
                        from .validation import aggregate
                        short=ensure_short(journal,p,accepted,protocol_path,prepared,execution)
                        result=aggregate({a:accepted['full_'+a] for a in ARMS},p,execution,sha(protocol_path),
                            sha(prepared),short,binary_receipt()['binary_sha256'])
                        result['preparation_reuse']=dict(path=str(accepted['reuse_preparation']),sha256=sha(accepted['reuse_preparation']))
                        path=attempt/'summary.json';write(path,result);write(journal.output/'summary.json',result)
                        with (journal.output/'summary.csv').open('x',newline='') as stream:
                            fields=['arm','training_seconds','order_excluded_seconds','speedup_vs_freq','cpu_feature_bytes','gpu_feature_cache_bytes']
                            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                            for a in ARMS:writer.writerow(dict(arm=a,**{k:result['arms'][a][k] for k in fields[1:]}))
                journal.commit(key,path);accepted[key]=path
            reuse.check(accepted['reuse_preparation'],p,protocol_path,execution)
            require(verify()==execution,'Sources changed before completion')
            journal.save(stage='complete',passed=True,complete=True,finished_unix=time.time())
        except BaseException as exc:
            journal.fail(exc);raise
        finally:
            signal.signal(signal.SIGTERM,old_term);signal.signal(signal.SIGINT,old_int)
