"""Metadata-only completed CL CSC binding and independent memory scenarios."""
from pathlib import Path
from candidates.ukl_feature_binding_v13.plan import read_metadata, ident, digest

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/n0/digit/cl_prepare_v4/full_csc_resume_v1')
N,E=978408098,43552515567
GIB,PAGE=2**30,4096


def rounded(n):return (n+PAGE-1)//PAGE*PAGE


def csc_binding(source=SOURCE):
    source=Path(source)
    if source.resolve(strict=True)!=source:raise ValueError('Canonical CL source required')
    state,receipt=read_metadata(source/'status.json')
    meta,meta_receipt=read_metadata(source/'csc.json')
    if (state.get('complete') is not True or state.get('full_csc') is not True
            or state.get('stage')!='complete' or state.get('nodes')!=N or state.get('edges')!=E
            or meta.get('complete') is not True or meta.get('orientation')!='incoming'
            or meta.get('nodes')!=N or meta.get('edges')!=E
            or meta.get('second_adjacency') is not False or meta.get('eid_array') is not False):
        raise ValueError('Completed original incoming CL CSC receipt required')
    specs={}
    for name,size in [('indptr',8*(N+1)),('indices',4*E)]:
        file=name+('.i64' if name=='indptr' else '.i32');p=source/file
        info=state['files'][file];h=info.get('sha256','');s=p.stat()
        if (p.resolve(strict=True)!=p or not p.is_file() or s.st_size!=size or info.get('bytes')!=size
                or len(h)!=64 or any(c not in '0123456789abcdef' for c in h)):
            raise ValueError('CL CSC identity/length/hash declaration mismatch')
        specs[name]=dict(path=str(p),offset=0,length=size,identity=ident(s),sha256=h)
    if len({s['identity'][0] for s in specs.values()})!=1:
        raise ValueError('CSC files must share the declared source device')
    result=dict(nodes=N,edges=E,specs=specs,receipts=[receipt,meta_receipt],
                logical_bytes=sum(v['length'] for v in specs.values()),
                arena_bytes=sum(rounded(v['length']) for v in specs.values()),
                full_payload_rehashed=False,graph_opened=False)
    result['binding_sha256']=digest(result)
    return result


def budget():
    # Overlay counts remain estimates until actual CL g2 receipts exist.
    primary=(N-N//10)//2;replicas=(N//5)//2;groups=primary+replicas
    raw={'indptr':8*(N+1),'indices':4*E}
    overlay={'group_ptr':8*(N+1),'group_ids':4*groups,'covered':16*groups,
             'members':8*groups,'bases':8*groups,'primary':8*N}
    csc=sum(rounded(v) for v in raw.values())
    grouped=csc+sum(rounded(v) for v in overlay.values())
    headroom=12*GIB;reserve=64*GIB;cache=128*2**20
    return dict(nodes=N,edges=E,csc_components_bytes=raw,overlay_scenario_bytes=overlay,
                csc_arena_bytes=csc,csc_memory_max=csc+headroom,
                csc_host_available_min=csc+headroom+reserve+cache,
                grouped_graph_scenario_bytes=grouped,
                grouped_memory_max_scenario=grouped+headroom,
                grouped_host_available_min_scenario=grouped+headroom+reserve+cache,
                primary_groups_scenario=primary,replica_groups_scenario=replicas,
                cpu_feature_cache_bytes=(N//10)*512,cpu_slot_map_bytes=4*N,
                application_read_bytes_per_second=512*2**20,hard_read_bytes_per_second=GIB,
                file_cache_limit_bytes=cache,host_reserve_bytes=reserve,headroom_bytes=headroom,
                full_load_enabled=False,gpu_enabled=False,formal_admission=False,
                limitations=['Only CSC sizes are actual; CL overlay does not yet exist',
                    'Training needs separate CPU hot cache, model and GPU memory budgets',
                    'Grouped EID resolution still needs high-degree performance evaluation'],
                required_controller_guards=['per-task cgroup memory high/max and swap=0',
                    'remaining-allocation memory admission','bounded O_DIRECT reads with pacing',
                    'kernel warning and writeback progress guard, Dirty and PSI checks',
                    'own file-cache limit','prewarm imports before ownership and CUDA',
                    'MADV_DONTFORK plus active no-child ownership guard',
                    'independent observer and post-release checks'])
