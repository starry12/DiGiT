"""Bounded source audit: headers, stat, properties and tiny sample reads only."""
import ast,hashlib,math,stat,struct
from pathlib import Path
from .common import cfg,source_root,identity,available_host

def header(path):
    path=Path(path)
    if not stat.S_ISREG(path.stat().st_mode):raise ValueError('Expected regular source file')
    with path.open('rb') as f:
        prefix=f.read(8)
        if prefix[:6]!=b'\x93NUMPY':raise ValueError('Expected NPY header')
        version=tuple(prefix[6:])
        if version==(1,0):size=struct.unpack('<H',f.read(2))[0]
        elif version==(2,0):size=struct.unpack('<I',f.read(4))[0]
        else:raise ValueError('Unsupported NPY version')
        if size>65536:raise ValueError('Oversized NPY header')
        value=ast.literal_eval(f.read(size).decode('latin1').strip());offset=f.tell()
    shape=value['shape'];dtype=value['descr']
    if not isinstance(shape,tuple) or any(type(x)!=int or x<0 for x in shape):raise ValueError('Bad NPY shape')
    item={'<i8':8,'<f4':4}.get(dtype)
    if item is None:raise ValueError('Unexpected source dtype')
    expected=offset+math.prod(shape)*item
    if expected!=path.stat().st_size:raise ValueError('NPY length mismatch')
    return dict(shape=list(shape),dtype=dtype,fortran_order=value['fortran_order'],header_bytes=offset,file_bytes=expected)

def audit(source=None,protocol=None):
    p=protocol or cfg();src=Path(source or source_root());edge=src/'edge_index.npy';prop=src/(p['upstream']+'.properties')
    before=identity(edge);h=header(edge);props={}
    for line in prop.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key,value=line.split('=',1);props[key]=value
    n,e=p['nodes'],p['source_edges']
    if int(props['nodes'])!=n or int(props['arcs'])!=e:raise ValueError('Properties differ from UKS contract')
    if h['shape']!=[2,e] or h['dtype']!='<i8' or h['fortran_order']:raise ValueError('Expected UKS C-order int64 [2,E]')
    samples=[]
    positions=sorted(set((i*(e-1)//23 for i in range(24)))) if e else []
    with edge.open('rb') as f:
        for pos in positions:
            f.seek(h['header_bytes']+pos*8);a=struct.unpack('<q',f.read(8))[0]
            f.seek(h['header_bytes']+(e+pos)*8);b=struct.unpack('<q',f.read(8))[0]
            if not (0<=a<n and 0<=b<n):raise ValueError('Sampled endpoint outside graph')
            samples.append(dict(edge_position=pos,src=a,dst=b))
    if identity(edge)!=before:raise ValueError('Source changed during audit')
    legacy={};labels=src/'node_label.npy'
    if labels.exists():
        lh=header(labels);values=[]
        if lh['shape']!=[n,1] or lh['dtype']!='<f4':raise ValueError('Unexpected old label header')
        with labels.open('rb') as f:
            for at in sorted(set(i*(n-1)//23 for i in range(24))):
                f.seek(lh['header_bytes']+at*4);value=struct.unpack('<f',f.read(4))[0]
                values.append(dict(node=at,value=value if math.isfinite(value) else None,finite=math.isfinite(value)))
        legacy=dict(header=lh,samples=values,nonfinite_samples=sum(not v['finite'] for v in values),reuse=False,reason='Unknown label/split provenance; generate seeded finite labels for actual backward passes')
    return dict(schema='digit-uks-source-audit-v1',source=str(src),header_checks_passed=True,full_source_validated=False,edges=dict(identity=before,header=h,samples=samples,full_sha256=None),properties=dict(path=str(prop),sha256=hashlib.sha256(prop.read_bytes()).hexdigest(),nodes=n,edges=e),legacy_labels=legacy,feature_file_exists=(src/'node_feat.npy').exists(),raw_payload_bytes_read=len(samples)*16+len(legacy.get('samples',[]))*4,bulk_scan=False,raw_ssd_access=False)

def capacity(protocol=None):
    p=protocol or cfg();n,e,d,k=p['nodes'],p['source_edges'],p['feature_dim'],p['cpu_cache_rows'];upper=e+n
    primary=(n-k)//2;replica=int(n*p['replication_ratio'])//2
    storage_upper=n+2*primary+4*replica+8
    return dict(nodes=n,source_edges=e,normalized_edges_upper=upper,actual_selfloops_unknown=True,
        raw_feature_payload_bytes=n*d*4,synthetic_labels_bytes=n*8,training_split_bytes=p['training']['train_nodes']*8,
        original_csc_runtime_upper_bytes=8*(n+1+2*upper),original_csc_file_upper_bytes=8*(n+1+upper)+256,
        cpu_feature_cache_bytes=k*d*4,reordered_feature_payload_upper_bytes=storage_upper*d*4,
        storage_row_upper=storage_upper,reordered_graph_file_upper_bytes=8*(n+primary+replica+1+upper),
        external_csc_scratch_estimate_bytes=48*upper+8*(3*n+e+1),host_required_bytes=p['host_required_bytes'],host_available_bytes=available_host(),
        gpu_admission_passed=False,gpu_budget_pending_actual_bundle=True,ssd_offsets=None,raw_ssd_payload_ready=False,
        caveat='Bounds for planning, not measured peaks or a whole-pipeline disk reservation. Other live jobs, grouping scratch and write amplification are additional.')
