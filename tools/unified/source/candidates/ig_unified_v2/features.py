from stage_profile import call as profile_call
from candidates.ig_unified_v2.common import *
L=DATA;CFG=ROOT/"configs/igb";K=cfg()["cpu_cache_rows"]
from ae.common import binding
import numpy as np,torch
import IGPerfNative as native
from ssd_guard import device_idle,require_payload
from io_accounting import reconcile
from candidates.io_accounting_v1.accounting import decode,useful_interval,validate_region

class NativeFeatures:
    def __init__(self,arm,out,mark):
        require(os.geteuid()==0,'Native feature mapping requires root');verify();device_idle()
        self.hot_start=read(L/'full/metadata_ready.json')['hot_start'] if arm=='full' else None;self.feature_seconds=0.;self.arm=arm;self.out=out;self.mark=mark;self.plan=read(CFG/'ssd_plan.json')[arm];self.intervals=[];self.routing_records=[];self.routing_passed=False
        self.payload_receipt=require_payload(arm);require(self.payload_receipt['plan']==self.plan,'Payload plan mismatch')
        self.controller=native.GIDS_Controllers();self.controller.init_GIDS_controllers(1,1024,128,[0]);s=native.BAM_Feature_Store_float();self.store=s
        p=self.plan;s.init_controllers(self.controller,p['slot_bytes'],p['offset']//p['slot_bytes'],4096,p['rows']*1024,1,int(arm=='full'),4096 if arm=='full' else 0)
        if arm=='full':
            from digit.io_geometry import IOGeometry
            geo=IOGeometry.create(feature_row_bytes=4096,group_size=2,minimum_transfer_bytes=4096,target_request_bytes=8192).with_payload_offset(p['offset'])
            self.geometry=geo.semantic_sha256();s.set_mixed_io_geometry(True,4096,4096,2,2,3,self.geometry)
        s.set_device_io_stats(True);native.short_begin(s,K);self.mark('native_allocated',native=native.admission_info(s))
        self.saved_pointers={k:v for k,v in native.admission_info(s).items() if k.endswith('pointer')}
    def populate(self,raw_source):
        p=self.plan;h=hashlib.sha256();last=0
        if self.arm=='gids':
            cache=read(CFG/'gids_cache.json');require(cache['passed'] and cache['rows']==K and cache['bytes']==K*4096 and cache['source_sha256']==self.plan['file_sha256'] and cache['rows_binding']==binding(L/'gids_cpu_rows.npy'),'GIDS cache binding')
            require(Path(cache['path']).stat().st_size==cache['bytes'],'GIDS cache size changed');file=open(cache['path'],'rb',buffering=0);offset=0;expected=cache['sha256'];self.hot=np.load(L/'gids_cpu_rows.npy',mmap_mode='r')
        else:
            file=open(L/'full/reordered_features.npy','rb',buffering=0);self.hot_start=read(L/'full/metadata_ready.json')['hot_start'];offset=p['header_bytes']+self.hot_start*4096;expected=read(CFG/'full_cache.json')['sha256'];self.hot=None
        for at in range(0,K,16384):
            hi=min(K,at+16384);blob=os.pread(file.fileno(),(hi-at)*4096,offset+at*4096);require(len(blob)==(hi-at)*4096,'Short CPU cache file')
            values=np.frombuffer(blob,dtype='<f4').reshape(-1,1024);require(np.isfinite(values).all(),'Nonfinite cache')
            rows=np.asarray(self.hot[at:hi],dtype=np.int64) if self.arm=='gids' else np.arange(self.hot_start+at,self.hot_start+hi,dtype=np.int64)
            native.short_fill(self.store,rows,values);h.update(blob)
            if time.time()-last>10 or hi==K:self.mark('cpu_cache_populate',rows_done=hi,total_rows=K);last=time.time()
        file.close();require(h.hexdigest()==expected and native.short_verify_map(self.store)==0,'Cache all-byte SHA/full map mismatch');self.cache_sha=h.hexdigest();self.mark('cpu_cache_verified',cache_sha256=self.cache_sha)
    def fetch(self,rows,flags):
        rows=np.ascontiguousarray(rows,dtype=np.int64);flags=np.ascontiguousarray(flags,dtype=np.bool_)
        require(rows.ndim==1 and flags.shape==rows.shape and 0<len(rows)<=16384 and np.all((rows>=0)&(rows<self.plan['rows'])),'Native request bounds')
        if self.arm=='gids':require(not flags.any(),'GIDS group request')
        elif flags.any():require(np.all(rows[flags]<self.hot_start),'Group request includes hot region')
        ids=torch.from_numpy(rows).cuda();groups=torch.from_numpy(flags).cuda();out=torch.empty((len(rows),1024),device='cuda')
        began=time.perf_counter();self.store.read_feature(out.data_ptr(),ids.data_ptr(),len(rows),1024,1024,0,groups.data_ptr());self.feature_seconds+=time.perf_counter()-began;return out
    def begin(self):
        torch.cuda.synchronize();self.store.reset_device_io_stats()
        return dict(feature=self.store.get_feature_access_stats(),gpu=self.store.get_gpu_cache_stats(),mixed=self.store.get_mixed_io_stats(),useful=decode(list(self.store.get_useful_io_stats())),feature_seconds=self.feature_seconds)
    def finish(self,before,count):
        torch.cuda.synchronize();feature=[b-a for a,b in zip(before['feature'],self.store.get_feature_access_stats())];gpu=self.store.get_gpu_cache_stats();mixed=self.store.get_mixed_io_stats();device=[int(x) for x in self.store.get_device_io_stats()]
        g=[b-a for a,b in zip(before['gpu'],gpu)];m=[b-a for a,b in zip(before['mixed'],mixed)]
        expected_commands=m[3]+m[4]+m[5] if self.arm=='full' else None
        expected_bytes=m[6]+m[7] if self.arm=='full' else None
        io=reconcile(device,expected_commands,expected_bytes)
        snapshot=dict(time=time.time(),rows=count,arm=self.arm,routing_complete=self.routing_passed,feature=feature,gpu_before=before['gpu'],gpu_after=gpu,gpu_delta=g,mixed_before=before['mixed'],mixed_after=mixed,mixed_delta=m,device_accounting=io)
        append_sync(self.out/'read_counter_snapshots.jsonl',snapshot)
        require(sum(feature)==count and min(feature)>=0,'Native feature count reconciliation; raw counters saved')
        require(len(gpu)==10 and gpu[0]==int(self.arm=='full') and gpu[1]==1048576 and gpu[2]==gpu[3]+gpu[4]+gpu[8] and gpu[9]==gpu[5]+gpu[6] and gpu[6]<=gpu[1],'GPU cache counters/policy/budget')
        if self.arm=='full':require(gpu[7]==gpu[9],'FIFO ticket mismatch')
        require(g[2]==feature[1],'GPU requests/feature routes mismatch')
        require(io['passed'],'Device primary/replay reconciliation; raw counters saved')
        if self.arm=='full':
            require(m[1]+m[2]==feature[1] and m[6]+m[7]==m[12]==io['primary_bytes'],'Mixed/primary byte reconciliation')
            require(mixed[13:]==[4096,4096,2,2,8192,3,4096],'Full runtime geometry mismatch')
        else:require(io['primary_bytes']==io['primary_commands']*4096,'GIDS raw primary geometry mismatch')
        return dict(useful=useful_interval(before['useful'],decode(list(self.store.get_useful_io_stats())),dict(cpu=feature[0],gpu_ssd=feature[1]),4096),feature_seconds=self.feature_seconds-before['feature_seconds'],rows=count,feature_cpu=feature[0],feature_gpu_ssd=feature[1],gpu_requests=g[2],gpu_hits=g[3],gpu_inserts=g[4],gpu_partial_fills=g[8],device_commands=device[3],device_bytes=device[4],primary_commands=io['primary_commands'],primary_bytes=io['primary_bytes'],replay_commands=device[9],replay_bytes=device[10],mixed=m[1:13] if self.arm=='full' else None,device_raw=device,gpu_after=gpu)
    def record_interval(self,stage,batch,record):
        self.intervals.append(dict(stage=stage,batch=batch,**record));write(self.out/'io_intervals.json',self.intervals)
    def routing(self,raw_source,arrays):
        def run(name,rows,group=False):
            rows=np.ascontiguousarray(rows,dtype=np.int64);before=self.begin();got=self.fetch(rows,np.full(len(rows),group,dtype=np.bool_));record=self.finish(before,len(rows))
            logical=rows if arrays is None else np.asarray(arrays['storage_to_node'][rows],dtype=np.int64)
            equal=bool(np.array_equal(got.cpu().numpy().view('u4'),raw_source.rows(logical).view('u4')))
            self.routing_records.append(dict(name=name,physical_rows=rows.tolist(),feature_sha256=digest(got),source_equal=equal,**record));write(self.out/'routing_partial.json',self.routing_records)
            require(equal,'Routing/source feature mismatch '+name);return record
        hot=np.asarray(self.hot[:64],dtype=np.int64) if self.arm=='gids' else np.arange(self.hot_start,self.hot_start+64,dtype=np.int64)
        a=run('mapped_cpu_fallback',hot);require(a['feature_cpu']==64 and a['device_commands']==0,'Mapped CPU fallback failed')
        if self.arm=='gids':
            candidates=np.arange(65536,dtype=np.int64);positions=np.searchsorted(self.hot,candidates);valid=positions<K;mask=np.ones(len(candidates),dtype=np.bool_);mask[valid]=self.hot[positions[valid]]!=candidates[valid];cold=candidates[mask][:64];require(len(cold)==64,'No cold routing candidates')
        else:cold=np.asarray(arrays['group_storage_base'][:64],dtype=np.int64)
        a=run('cold_raw_first',cold);require(a['feature_cpu']==0 and a['primary_bytes']==64*4096 and a['primary_commands']==64,'Raw SSD route failed')
        a=run('cold_raw_repeat',cold);require(a['gpu_hits']==64 and a['device_commands']==0,'Raw GPU repeat failed')
        if self.arm=='full':
            groups=np.asarray(arrays['group_storage_base'][64:128],dtype=np.int64)
            a=run('cold_group_first',groups,True);require(a['primary_bytes']==64*8192 and a['primary_commands']==64,'Full group SSD8KiB route failed')
            a=run('cold_group_repeat',groups,True);require(a['gpu_hits']==64 and a['device_commands']==0,'Group GPU repeat failed')
            base=int(arrays['group_storage_base'][128]);a=run('partial_raw_first',[base]);require(a['primary_bytes']==4096 and a['primary_commands']==1,'Partial raw first')
            a=run('partial_group_extend',[base],True);require(a['primary_bytes']==4096 and a['primary_commands']==1,'Group companion4KiB extension')
            a=run('partial_companion_repeat',[base+1]);require(a['device_commands']==0 and a['gpu_hits']==1,'Companion repeated hit')
            a=run('odd_hot_tail',[self.hot_start+K-1]);require(a['feature_cpu']==1 and a['device_commands']==0,'Odd hot tail CPU map')
        # A routing-only forced SSD load seeds a mapped hot row into the real GPU cache.
        # The normal mixed/legacy reader must then prefer its resident GPU copy.
        ids=torch.tensor([int(hot[0])],dtype=torch.int64,device='cuda');flags=torch.zeros(1,dtype=torch.bool,device='cuda');out=torch.empty((1,1024),device='cuda');before=self.begin()
        native.short_prime(self.store,out.data_ptr(),ids.data_ptr(),1,flags.data_ptr());record=self.finish(before,1)
        logical=np.array([hot[0]],dtype=np.int64) if arrays is None else np.array([arrays['storage_to_node'][hot[0]]],dtype=np.int64)
        require(np.array_equal(out.cpu().numpy().view('u4'),raw_source.rows(logical).view('u4')) and record['primary_bytes']==4096 and record['primary_commands']==1,'Routing prime source/SSD')
        self.routing_records.append(dict(name='force_ssd_hot_for_routing',**record));a=run('mapped_hot_gpu_first',[hot[0]])
        require(a['feature_cpu']==0 and a['feature_gpu_ssd']==1 and a['gpu_hits']==1 and a['device_commands']==0,'Mapped hot GPU-first failed')
        self.routing_passed=True;write(self.out/'routing.json',dict(passed=True,arm=self.arm,cases=self.routing_records,hot_priming_scope='routing-only forced native SSD read; normal policy used for measured probe/training'))
    def final_checks(self):
        require(native.short_verify_map(self.store)==0,'Final CPU page map changed');info=native.admission_info(self.store)
        require(self.saved_pointers=={k:v for k,v in info.items() if k.endswith('pointer')} and info['cpu_loaded_rows']==K and info['cpu_cache_bytes']==K*4096 and info['gpu_cache_bytes']==4*2**30,'Native allocation changed')
        require(len(self.intervals)==50 and any(x['feature_cpu']>0 for x in self.intervals if x['stage']=='train') and any(x['device_commands']>0 for x in self.intervals if x['stage']=='train'),'Training route coverage missing')
    def receipt(self):
        return dict(passed=True,arm=self.arm,cache_rows=K,cache_sha256=self.cache_sha,routing_passed=self.routing_passed,routing=binding(self.out/'routing.json'),intervals=binding(self.out/'io_intervals.json'),counter_snapshots=binding(self.out/'read_counter_snapshots.jsonl'),payload_provenance=self.payload_receipt,native_info=native.admission_info(self.store))

# Preserve accepted allocation/prefill/read and primary/replay code. Production
# only invokes routing() during smoke, with window-based counters in all runs.
class LegacyFormalFeatures(NativeFeatures):
    def __init__(self,arm,out,mark):
        super().__init__(arm,out,mark)
        self.hot_start=read(L/'full/metadata_ready.json')['hot_start'] if arm=='full' else None
    def final_checks(self):
        require(native.short_verify_map(self.store)==0,'Final CPU map changed')
        info=native.admission_info(self.store)
        require(self.saved_pointers=={k:v for k,v in info.items() if k.endswith('pointer')}
                and info['cpu_loaded_rows']==K and info['gpu_cache_bytes']==4*2**30,'Resident allocation changed')
    def receipt(self):
        return dict(passed=True,arm=self.arm,cache_rows=K,cache_sha256=self.cache_sha,
                    route_warmup=self.routing_passed,payload_provenance=self.payload_receipt,native_info=native.admission_info(self.store),
                    interval_scope='100 measured batches or warmup/smoke phase tail; CPU/GPU/SSD counters, no source comparison')

class FormalFeatures(LegacyFormalFeatures):
    """UKS-style aliases and device requests, retaining IG real 4K/8K features."""
    def set_hot(self,hot,arrays):
        require(self.arm=='full' and hot.shape==(K,) and np.all(hot[1:]>hot[:-1]),'Invalid IG logical hot set')
        self.logical_hot=hot;self.primary=arrays['node_to_primary_row']
        self.physical_hot=np.sort(np.asarray(self.primary[hot],dtype=np.int64))
        require(np.all(self.physical_hot[1:]>self.physical_hot[:-1]),'Primary feature addresses not unique')
        self.inverse=arrays['storage_to_node']

    def populate(self,raw_source):
        require(native.IG_ASYNC_ABI==1,"Missing local async ABI")
        require(native.OPTIMIZED_IG_ABI==1 and native.DEMAND_PAIR_ABI==1,'Wrong demand-pair native ABI')
        if self.arm!='full':return super().populate(raw_source)
        from .cache import accepted_hot
        from .preload import populate
        from candidates.ig_demand_pair_v1.mapping import alias_chunks
        hot,_,expected=accepted_hot(self.profile_directory,self.profile_config)
        require(np.array_equal(hot,self.logical_hot),'Preload hot set mismatch')
        require(np.array_equal(np.sort(self.inverse[self.physical_hot]),hot),'Invalid primary inverse')
        self.physical_hot_digest=digest(self.physical_hot)
        self.source_path=L/'full/reordered_features.npy'
        self.pack_receipt=populate(self,expected)
        mapped=0
        for lo,slots in alias_chunks(self.physical_hot,self.inverse,len(self.primary)):
            native.alias_chunk(self.store,lo,slots);mapped+=int(np.count_nonzero(slots))
        native.alias_finish(self.store)
        self.alias_rows=mapped
        self.mark('cpu_cache_verified',cache_sha256=self.cache_sha,alias_rows=mapped,pack=self.pack_receipt)

    def fetch_device(self,rows,flags,asynchronous=False):
        require(rows.is_cuda and flags.is_cuda and rows.device==flags.device and
                rows.dtype==torch.int64 and flags.dtype==torch.bool and rows.ndim==1 and
                rows.shape==flags.shape and 0<len(rows)<=2097152,'Wrong device request tensors')
        rows=rows.contiguous();flags=flags.contiguous()
        out=torch.empty((len(rows),1024),device=rows.device)
        stream=torch.cuda.current_stream(rows.device)
        began=time.perf_counter()
        table_ptr=capacity=filtered_ptr=0
        if self.arm=='full':
            # Reuse bounded scratch; includes this work in feature timing.
            capacity=1<<(2*len(rows)-1).bit_length()
            if not hasattr(self,'demand_table') or self.demand_table.numel()<capacity:
                self.demand_table=torch.empty(capacity,dtype=torch.int64,device=rows.device)
                self.demand_flags=torch.empty(capacity//2,dtype=torch.bool,device=rows.device)
            table_ptr=self.demand_table.data_ptr();filtered_ptr=self.demand_flags.data_ptr()
        (native.device_submit if asynchronous else native.device_read)(self.store,out.data_ptr(),rows.data_ptr(),flags.data_ptr(),len(rows),
                           self.hot_start or 0,stream.cuda_stream,table_ptr,capacity,filtered_ptr)
        self.feature_seconds+=time.perf_counter()-began
        return out

    def fetch(self,rows,flags):
        # CPU arrays are only used by the bounded routing smoke.
        rows=np.ascontiguousarray(rows,dtype=np.int64);flags=np.ascontiguousarray(flags,dtype=np.bool_)
        return self.fetch_device(torch.from_numpy(rows).cuda(),torch.from_numpy(flags).cuda())

    def normalize(self,*args):
        raise RuntimeError('Training must not normalize requests through a CPU hot lookup')

    def routing(self,raw_source,arrays):
        if self.arm!='full':return LegacyFormalFeatures.routing(self,raw_source,arrays)
        cases=[]
        def run(name,rows,group=False):
            rows=np.ascontiguousarray(rows,dtype=np.int64);before=self.begin()
            got=self.fetch(rows,np.full(len(rows),group,dtype=np.bool_));r=self.finish(before,len(rows))
            require(np.array_equal(got.cpu().numpy().view('u4'),raw_source.rows(self.inverse[rows]).view('u4')),
                    'Native/source alias mismatch: '+name)
            cases.append(dict(name=name,physical_rows=rows.tolist(),source_equal=True,**r));return r
        r=run('new_hot_cpu',self.physical_hot[:64]);require(r['feature_cpu']==64 and not r['device_commands'],'Hot CPU route')
        bases=np.asarray(arrays['group_storage_base'][:1048576],dtype=np.int64)
        logical=np.asarray(self.inverse[np.concatenate([bases,bases+1])],dtype=np.int64)
        pos=np.searchsorted(self.logical_hot,logical);hit=pos<K
        hit[hit]=self.logical_hot[pos[hit]]==logical[hit]
        replicas=np.concatenate([bases,bases+1])[hit][:64]
        require(len(replicas)==64,'Insufficient mapped replicas for smoke')
        r=run('replica_alias_cpu',replicas,True)
        require(r['feature_cpu']==64 and not r['device_commands'],'Replica CPU alias route')
        cold=bases[~(hit[:len(bases)]|hit[len(bases):])][:66]
        require(len(cold)==66,'Insufficient truly cold group pairs')
        pair_rows=np.stack([cold[:64],cold[:64]+1],axis=1).reshape(-1)
        r=run('cold_group_first',pair_rows,True);require(r['primary_bytes']==64*8192 and r['primary_commands']==64,'Demanded 8K route')
        r=run('cold_group_repeat',cold[:64],True);require(r['gpu_hits']==64 and not r['device_commands'],'Group GPU repeat')
        r=run('unpaired_group',cold[64:65],True);require(r['primary_bytes']==4096 and r['primary_commands']==1,'Unrequested companion must not be read')
        r=run('unpaired_resident',cold[64:65],True);require(r['gpu_hits']==1 and not r['device_commands'],'Resident row must not fill absent sibling')
        r=run('partial_group_extend',[cold[64],cold[64]+1],True);require(r['primary_bytes']==4096 and r['primary_commands']==1,'Only missing demanded row is fetched')
        r=run('partial_repeat',cold[64:65]+1);require(r['gpu_hits']==1 and not r['device_commands'],'Companion hit')
        self.routing_passed=True;write(self.out/'routing.json',dict(passed=True,arm=self.arm,cases=cases))

    def receipt(self):
        r=LegacyFormalFeatures.receipt(self)
        r.update(request_path='validated CUDA addresses, bounded native launches directly into final output',
                 feature_seconds_scope='native request guard and read kernels including stream completion')
        r['pair_read_policy']='exact batch physical counterpart, neither CPU-mapped; locked GPU state determines 0/4/8KiB, requested-row hit never refills sibling'
        if self.arm=='full':r.update(cpu_hot_policy='full_epoch_frequency',
            replica_alias_policy='all physical replicas map to one native CPU slot; GPU-first retained',
            logical_hot_sha256=digest(self.logical_hot),physical_hot_sha256=digest(self.physical_hot),
            alias_rows=self.alias_rows,pack=self.pack_receipt)
        return r


def aggregate(windows):
    require(windows,'Missing phase I/O windows')
    region=windows[0]['useful']['region_id']
    require(all(w['useful']['region_id']==region for w in windows),'Phase counter leakage')
    device=dict(enabled=True,reconciled=True,submitted_commands=sum(w['device_raw'][2] for w in windows),completed_commands=sum(w['device_raw'][3] for w in windows),
                completed_bytes=sum(w['device_bytes'] for w in windows),primary_bytes=sum(w['primary_bytes'] for w in windows),replay_bytes=sum(w['replay_bytes'] for w in windows),active_ns=sum(w['device_raw'][5] for w in windows))
    useful=dict(schema='digit-useful-io-v1',enabled=True,reconciled=True,region_id=region,granularity_bytes=512,
                **{k:sum(w['useful'][k] for w in windows) for k in ('ssd_fill_bytes','ssd_useful_bytes','gpu_feature_bytes','gpu_feature_rows')})
    value=dict(device=device,useful_io=useful,feature=dict(cpu=sum(w['feature_cpu'] for w in windows),gpu_ssd=sum(w['feature_gpu_ssd'] for w in windows)),
               feature_row_bytes=4096,feature_seconds=sum(w['feature_seconds'] for w in windows))
    validate_region(value);return value
