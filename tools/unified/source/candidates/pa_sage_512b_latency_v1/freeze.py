"""Freeze source and exact DGL internals after independent and combined checks."""
from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Already frozen')
    base=baseline.verify();check_inputs(read(OUT/'inputs.json'))
    for name in ('cpu_checks.json','gpu_checks.json'):
        result=read(OUT/name);require(result['passed'],'Checks failed')
        for rel,digest in result['tested_sha256'].items():require(sha(ROOT/rel)==digest,'Tested source changed: '+rel)
    setup()
    import dgl.convert
    import dgl.heterograph
    import dgl.ops.spmm
    import dgl.nn.pytorch.conv.sageconv
    import dgl.backend.pytorch.sparse
    import torch.optim.adam
    import importlib
    modules=['dgl.convert','dgl.heterograph','dgl.ops.spmm','dgl.nn.pytorch.conv.sageconv',
             'dgl.backend.pytorch.sparse','torch.optim.adam','models']
    runtime=[importlib.import_module(name).__file__ for name in modules]
    write(HERE/'manifest.json',dict(schema='pa-sage-512b-latency-v1',baseline_sha256=base,
        files={str(p.relative_to(ROOT)):sha(p) for p in source_files()},created_unix=time.time(),
        runtime_sha256={p:sha(p) for p in runtime},
        evidence_sha256={name:sha(OUT/name) for name in ('inputs.json','cpu_checks.json','gpu_checks.json')}))
    print('Frozen independent latency candidate:',verify(),flush=True)


if __name__=='__main__':main()
