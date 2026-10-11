"""GPU attribution without discarding journal records or host-wide protection."""
import re
from candidates.ukl_native_sampling_v10r4.guard import KernelCursor as Base

PCI = re.compile(r'(?<![0-9a-f])(?:([0-9a-f]{4,8}):)?([0-9a-f]{2}):([0-9a-f]{2})(?:\.([0-7]))?(?![0-9a-f])', re.I)
UUID = re.compile(r'GPU-[0-9a-f]{8}-[0-9a-f-]{27,}', re.I)
TIMEOUT = re.compile(r'^NVRM: _threadNodeCheckTimeout: (?:_threadNodeCheckTimeout: )?(?:currentTime: [0-9a-fx]+ >= [0-9a-fx]+|Timeout was set to: 4000 msecs!)$', re.I)
SYSTEM = re.compile(r'out of memory|oom[-_ ]kill|kernel panic|BUG:|Oops:|soft lockup|hard LOCKUP|hung task|blocked for more than|I/O error|EXT4-fs error|nvme.*(?:timeout|reset|error)|Call Trace:', re.I)

def pci_key(value):
    match = PCI.fullmatch(value.strip())
    if not match:raise ValueError('Invalid PCI address')
    domain,bus,device,function=match.groups()
    return (int(domain or '0',16),int(bus,16),int(device,16),int(function or '0'))

class KernelCursor(Base):
    def __init__(self,*args,selected_uuid,selected_pci,**kwargs):
        super().__init__(*args,**kwargs)
        if not UUID.fullmatch(selected_uuid):raise ValueError('Invalid selected GPU UUID')
        self.selected_uuid=selected_uuid.lower();self.selected_pci=pci_key(selected_pci)
        self.gpu_classifications=[]

    def _classify(self,records):
        candidates=super()._classify(records)
        self.gpu_classifications=[];fatal=[]
        for record in candidates:
            msg=record['MESSAGE'];reason='system_or_unattributed_alert';stop=True
            if not SYSTEM.search(msg) and msg.startswith(('NVRM:','nvidia ')):
                uuids={s.lower() for s in UUID.findall(msg)}
                addresses={pci_key(m.group()) for m in PCI.finditer(msg)}
                if self.selected_uuid in uuids or self.selected_pci in addresses:
                    reason='selected_gpu'
                elif uuids or addresses:
                    reason='other_gpu';stop=False
                elif TIMEOUT.fullmatch(msg):
                    reason='unattributed_thread_timeout';stop=False
            self.gpu_classifications.append(dict(cursor=record['__CURSOR'],reason=reason,fatal=stop,message=msg))
            if stop:fatal.append(record)
        return fatal

    def _result(self,query_ok,records,alerts,current_error=None):
        result=super()._result(query_ok,records,alerts,current_error)
        result.update(gpu_scope=dict(uuid=self.selected_uuid,pci=list(self.selected_pci)),
                      gpu_alert_classification=self.gpu_classifications if records else [])
        return result
