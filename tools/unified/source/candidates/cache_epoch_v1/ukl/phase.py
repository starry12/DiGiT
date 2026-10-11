"""One-way init -> data accounting within a single bounded systemd service."""
import json
import os
from pathlib import Path
import time

GIB=2**30
MIB=2**20
INIT_LIMIT=GIB
DATA_LIMIT=128*MIB
PREFIX='/system.slice/digit-ukl-common-gpu-v27-'
CGROOT=Path('/sys/fs/cgroup')


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(str(tmp),str(path))


def membership(pid):
    return next(s[3:] for s in Path('/proc/%d/cgroup'%pid).read_text().splitlines() if s.startswith('0::'))


def start_ticks(pid):
    return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]


def await_phase(out, target, check=lambda:None):
    if target not in ('init','data'):raise ValueError('Unknown phase')
    pid=os.getpid();ticks=start_ticks(pid)
    write(out/'phase_request.json',dict(pid=pid,start_ticks=ticks,target=target))
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        check()
        if (out/'STOP').exists():raise RuntimeError('Stop during phase handoff')
        p=out/('phase_'+target+'.json')
        if p.exists():
            r=json.loads(p.read_text())
            if r['pid']!=pid or r['start_ticks']!=ticks or r['phase']!=target:
                raise RuntimeError('Phase acknowledgement identity mismatch')
            if membership(pid)!=r['cgroup']:
                raise RuntimeError('Actual cgroup differs from acknowledgement')
            return r
        time.sleep(.05)
    raise RuntimeError('Phase acknowledgement timeout')


def stats(path):
    values={k:int(v) for k,v in (l.split() for l in (path/'memory.stat').read_text().splitlines())}
    return {k:values[k] for k in ('file','anon','shmem','file_dirty','file_writeback')}


def create_probe(out):
    path=out/'phase_init_probe.bin'
    with path.open('xb') as stream:
        for _ in range(4):stream.write(bytes(MIB))
        stream.flush();os.fsync(stream.fileno())
    return path


def verify_charge_handoff(out, ack, init_probe):
    # No cache dropping or old-file manipulation. These are unique 4 MiB files
    # owned by this attempt and are removed before graph allocation.
    parent=CGROOT/str(Path(ack['cgroup']).parent).lstrip('/')
    path=out/'phase_data_probe.bin'
    try:
        with path.open('xb') as stream:
            for _ in range(4):stream.write(bytes(MIB))
            stream.flush();os.fsync(stream.fileno())
        initial,data=stats(parent/'init'),stats(parent/'data')
        if not (4*MIB<=initial['file']<=INIT_LIMIT and 4*MIB<=data['file']<=DATA_LIMIT):
            raise RuntimeError('Separate file-cache ownership probe failed')
        value=dict(passed=True,probe_bytes_each=4*MIB,init=initial,data=data,
                   phase_ack=ack,graph_allocated=False)
        write(out/'phase_charge_probe.json',value)
        return value
    finally:
        for own in (path,init_probe):
            if own.exists():own.unlink()


class Manager:
    def __init__(self, service, out):
        if not service.startswith(PREFIX) or not service.endswith('.service') or len(Path(service).parts)!=3:
            raise ValueError('Exact owned service path required')
        self.service,self.out,self.phase=service,out,'boot'
        self.root=CGROOT/service.lstrip('/')
        self.pid=self.ticks=None

    def poll(self,state):
        req=self.out/'phase_request.json'
        if not req.exists():return
        r=json.loads(req.read_text());target=r['target']
        if target==self.phase:return
        if (self.phase,target) not in (('boot','init'),('init','data')):
            raise RuntimeError('Invalid phase transition')
        pid=int(state.get('MainPID','0'))
        if pid<=0 or r['pid']!=pid or str(r['start_ticks'])!=start_ticks(pid):
            raise RuntimeError('Phase request does not match current service PID')
        source=self.root if self.phase=='boot' else self.root/'init'
        if membership(pid)!=(self.service if self.phase=='boot' else self.service+'/init'):
            raise RuntimeError('Worker outside expected source cgroup')
        if set((source/'cgroup.procs').read_text().split())!={str(pid)}:
            raise RuntimeError('Unexpected processes in phase cgroup')
        if self.phase=='boot':
            self.pid,self.ticks=pid,r['start_ticks']
            if not {'memory','cpu','io','pids'}<=set((self.root/'cgroup.controllers').read_text().split()):
                raise RuntimeError('Required phase controllers unavailable')
            (self.root/'init').mkdir()
            (self.root/'init/cgroup.procs').write_text(str(pid))
            if (self.root/'cgroup.procs').read_text().strip():
                raise RuntimeError('Service root not empty before enabling controllers')
            (self.root/'cgroup.subtree_control').write_text('+memory +cpu +io +pids')
            (self.root/'data').mkdir()
            for child in ('init','data'):
                leaf=self.root/child
                for name in ('memory.max','memory.high','memory.swap.max','pids.max','cpu.max'):
                    (leaf/name).write_text((self.root/name).read_text())
                for line in (self.root/'io.max').read_text().splitlines():
                    (leaf/'io.max').write_text(line)
        else:
            if (pid,r['start_ticks'])!=(self.pid,self.ticks):raise RuntimeError('PID changed during initialization')
            if (self.root/'data/cgroup.procs').read_text().strip():raise RuntimeError('Data cgroup already occupied')
            (self.root/'data/cgroup.procs').write_text(str(pid))
        from candidates.ukl_training_native_v15r11.cgroup_limits import verify_leaf
        for child in ('init','data'):verify_leaf(self.root/child,self.root)
        expected=self.service+'/'+target
        for task in (Path('/proc')/str(pid)/'task').iterdir():
            rows=(task/'cgroup').read_text().splitlines()
            if '0::'+expected not in rows:raise RuntimeError('Not all worker threads moved together')
        self.phase=target
        write(self.out/('phase_'+target+'.json'),dict(phase=target,pid=pid,start_ticks=self.ticks,
              cgroup=expected,service=self.service,initialization_file_limit=INIT_LIMIT,
              data_file_limit=DATA_LIMIT))

    def attach(self,sample):
        # Called by the parent controller; worker cannot choose accounting limits.
        if 'cgroup' not in sample:return sample
        sample['accounting_phase']=self.phase
        sample['phase_cgroups']={}
        if self.phase=='boot':return sample
        try:
            for child in ('init','data'):
                p=self.root/child
                sample['phase_cgroups'][child]=dict(memory=stats(p),
                    events={k:int(v) for k,v in (l.split() for l in (p/'memory.events').read_text().splitlines())})
        except (OSError,ValueError,KeyError) as exc:
            sample['errors'].append('phase_cgroups: '+str(exc))
        return sample
