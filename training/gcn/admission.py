"""Retain the accepted common budget and reserve GCN degree/normalization buffers."""
import math
from training.gcn.common import *

def estimate():
    from training.sage.admission import estimate as parent_estimate
    parent=parent_estimate();p=cfg();old=read(PARENT/'protocol.json')
    for key in ('fanouts','batch_size','hidden','classes','layers','gpu_cache_bytes','cpu_cache_rows','graph','base_layout','data','metadata_mode'):
        require(p[key]==old[key],'Shared budget/layout incompatibility: '+key)
    parts=dict(parent['components_bytes']);frontier=p['batch_size'];blocks=[]
    for fanout in reversed(p['fanouts']):
        src=min(p['graph']['nodes'],frontier*(fanout+1));blocks.insert(0,dict(src=src,dst=frontier,edges=frontier*fanout));frontier=src
    widths=[p['input_dim'],p['hidden'],p['hidden'],p['classes']]
    parameters=sum(widths[i]*widths[i+1]+widths[i+1] for i in range(p['layers']))
    require(parameters==p['model_parameter_count'],'GCN parameter envelope differs')
    parts['model_optimizer_grad']=max(parts['model_optimizer_grad'],parameters*32)
    parts['gcn_normalization_backward_allowance']=sum((b['src']+b['dst'])*(4*4+2*4*max(widths[i],widths[i+1])) for i,b in enumerate(blocks))
    return dict(parent,schema='digit-pa-gcn-admission-v1',model='gcn',components_bytes=parts,
                required_bytes=math.ceil(sum(parts.values())*1.2),exact_parameters=parameters,
                block_upper_envelope=blocks,protocol_sha256=sha(P),parent_required_bytes=parent['required_bytes'],
                formal_bound=False,reservation=False,
                note='Conservative estimate for both systems, not an OOM proof. Retains every PA/SAGE common reserve and adds GCN block degree/normalization forward-backward storage. Verify native peaks before full run.')
