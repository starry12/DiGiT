"""Require actual CUDA graph replay; kernel sums are diagnostic only."""
from .common import read, require, sha


def kernel_evidence(path, mode):
    events = read(path)['traceEvents']
    kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    runtime = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cuda_runtime']
    graphs = [e for e in runtime if e.get('name', '').startswith('cudaGraphLaunch')]
    launches = [e for e in runtime if e.get('name', '').startswith('cudaLaunchKernel')]
    require(bool(kernels), 'No CUDA kernels in trace')
    require(len(graphs) == (32 if mode == 'graph' else 0), 'Expected two graph replays per each of 16 full batches')
    return dict(mode=mode, trace_sha256=sha(path), graph_launches=len(graphs),
        ordinary_kernel_launches=len(launches), cuda_kernels=len(kernels),
        graph_launch_host_ms=sum(e.get('dur', 0) for e in graphs)/1000,
        kernel_launch_host_ms=sum(e.get('dur', 0) for e in launches)/1000,
        scope='16 profiled full batches; forward and backward graph launches; not E2E performance',
        performance_evidence=False)


def compare_traces(eager, graph):
    require(eager['mode'] == 'eager' and graph['mode'] == 'graph', 'Wrong trace mode')
    require(eager['graph_launches'] == 0 and graph['graph_launches'] == 32, 'Graph forward/backward replays absent')
    return dict(eager=eager, graph=graph,
        ordinary_launch_reduction=eager['ordinary_kernel_launches'] - graph['ordinary_kernel_launches'],
        limits='A short trace proves replay, not full-epoch speedup; host launch spans are not kernel times')
