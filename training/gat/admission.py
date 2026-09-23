"""Retain PA common reserves and explicitly account for four-head attention."""
import math
from training.gat.common import *

def estimate():
    from training.sage.admission import estimate as parent_estimate
    parent=parent_estimate();p=cfg();old=read(PARENT/'protocol.json')
    for key in ('fanouts','batch_size','hidden','classes','layers','gpu_cache_bytes','cpu_cache_rows','graph','base_layout','data','metadata_mode'):
        require(p[key]==old[key],'Shared budget/layout incompatibility: '+key)
    parts=dict(parent['components_bytes']);frontier=p['batch_size'];blocks=[]
    for fanout in reversed(p['fanouts']):
        src=min(p['graph']['nodes'],frontier*(fanout+1));blocks.insert(0,dict(src=src,dst=frontier,edges=frontier*fanout));frontier=src
    heads=p['model_config']['num_heads'];inputs=[128,heads*128,heads*128];outputs=[heads*128,heads*128,heads*172]
    parameters=sum(i*o+3*o for i,o in zip(inputs,outputs))
    require(parameters==p['model_parameter_count']==685072 and heads==4,'GAT parameter envelope differs')
    parts['model_optimizer_grad']=max(parts['model_optimizer_grad'],parameters*32)
    detail=[]
    for b,width in zip(blocks,outputs):
        detail.append(dict(projections_and_backward=4*4*(b['src']+b['dst'])*width,
                           weighted_message_allowance=4*b['edges']*width,
                           attention_scalars_and_backward=4*heads*(6*b['edges']+4*(b['src']+b['dst'])),
                           hidden_output_and_backward=4*4*b['dst']*width))
    parts['gat_attention_backward_allowance']=sum(sum(d.values()) for d in detail)
    envelope=sum(parts[k] for k in ('model_optimizer_grad','batch_features_activations','sampled_edges_and_sort_scratch','gat_attention_backward_allowance'))
    return dict(parent,schema='digit-pa-gat-admission-v1',model='gat',components_bytes=parts,
                required_bytes=math.ceil(sum(parts.values())*1.2),exact_parameters=parameters,
                block_upper_envelope=blocks,attention_allowances_per_block=detail,model_probe_limit_bytes=envelope,
                protocol_sha256=sha(P),parent_required_bytes=parent['required_bytes'],formal_bound=False,reservation=False,
                note='Engineering estimate, not an OOM proof. Retains all PA/SAGE common reserves. Adds four projected source/destination buffers, one potential edge-vector buffer, attention scalar forward/backward buffers and four hidden output buffers per block. DGL normally fuses weighted messages. Verify synthetic maximum-row training and native smoke peaks before full run.')
