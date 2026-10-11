"""Normalize the accepted CL stages without relabeling historical receipts."""
import json
from . import dataset as D

def accepted_inputs():
    C=D.C;C.verify_manifest();C.binding()
    saved=json.loads((C.OUT/'runtime_inputs.json').read_text())
    stages={s:C.accepted_stage(s) for s in C.STAGES}
    if not saved.get('passed') or saved['stages']!=stages or saved['source']!=C.metadata():
        raise RuntimeError('CL accepted preparation index changed')
    graph=dict(stages['graph'])
    graph['files']={**stages['graph']['files'],**stages['mapping']['files'],**stages['split']['files']}
    stages=dict(stages,graph=graph)
    return dict(prepared=True,dataset='CL',feature_dim=128,row_bytes=512,classes=19,
        feature_dtype='float32',nodes=D.N,storage_rows=D.ROWS,cpu_cache_rows=D.N//10,
        gpu_cache_bytes=4*2**30,stages=stages,source_index_sha256=C.sha(C.OUT/'runtime_inputs.json'))

def arm_inputs(value,arm):
    from candidates.ukl_training_adapter_v12.inputs import arm_inputs as base
    return base(value,arm)

def load_window(value):
    from candidates.ukl_training_adapter_v12.inputs import load_window as base
    return base(value)
