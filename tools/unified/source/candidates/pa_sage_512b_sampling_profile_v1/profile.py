"""Host-only nested spans. No CUDA events, device queries or synchronization."""
from contextlib import contextmanager
from functools import wraps
import importlib
import math
import time


class HostProfile:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.stack = []
        self.records = {}
        self.restores = []
        self.layer = None

    @contextmanager
    def span(self, name):
        start = self.clock()
        frame = [name, 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - start
            assert self.stack.pop() is frame
            path = '/'.join([f[0] for f in self.stack] + [name])
            self.records.setdefault(path, []).append((elapsed, elapsed - frame[1]))
            if self.stack:
                self.stack[-1][1] += elapsed

    def replace(self, obj, name, value):
        # Restore instance attributes by deleting overrides, not retaining bound methods.
        local = name in vars(obj)
        original = getattr(obj, name)
        self.restores.append((obj, name, original, local))
        setattr(obj, name, value)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def measured(*args, **kwargs):
            with self.span(label):
                return original(*args, **kwargs)
        self.replace(obj, name, measured)

    def close(self):
        if self.stack:
            raise RuntimeError('Unclosed profile spans')
        for obj, name, original, local in reversed(self.restores):
            if local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.restores.clear()

    def summary(self):
        if self.stack:
            raise RuntimeError('Cannot report incomplete spans')
        spans = {}
        for name, values in self.records.items():
            inclusive = [a for a, _ in values]
            ordered = sorted(inclusive)
            spans[name] = dict(calls=len(values), inclusive_seconds=sum(inclusive)/1e9,
                exclusive_seconds=sum(b for _, b in values)/1e9,
                first_seconds=inclusive[0]/1e9,
                subsequent_seconds=sum(inclusive[1:])/1e9,
                p50_us=ordered[(len(ordered)-1)//2]/1000,
                p95_us=ordered[math.ceil(len(ordered)*.95)-1]/1000,
                max_us=max(ordered)/1000)
        return dict(mode='host', spans=spans, added_cuda_synchronization=False,
            cuda_events=False, standalone_cuda_kernel_times=False,
            exclusive_note='Parent minus measured children; Python timing overhead remains',
            native_call_note='Host CUDA dispatch only; existing downstream waits remain at their original sites')


def install(profile, sampler, graph, loader, arm):
    """Process-local wrappers installed before DGL captures sampler.sample."""
    import dgl
    original_blocks = sampler.sample_blocks
    def blocks(*args, **kwargs):
        profile.layer = len(sampler.fanouts) - 1
        with profile.span('blocks'):
            return original_blocks(*args, **kwargs)
    profile.replace(sampler, 'sample_blocks', blocks)
    profile.wrap(sampler, 'sample', 'sampler')
    original_neighbors = graph.sample_neighbors
    def neighbors(*args, **kwargs):
        with profile.span('layer%d.dgl_neighbors' % profile.layer):
            return original_neighbors(*args, **kwargs)
    profile.replace(graph, 'sample_neighbors', neighbors)
    target = dgl if arm == 'digit' else importlib.import_module(type(sampler).__module__)
    original_to_block = target.to_block
    def to_block(*args, **kwargs):
        with profile.span('layer%d.to_block' % profile.layer):
            result = original_to_block(*args, **kwargs)
        profile.layer -= 1
        return result
    profile.replace(target, 'to_block', to_block)
    if arm == 'digit':
        profile.wrap(sampler, 'sample_outer_frontier', 'outer')
        profile.wrap(sampler, 'build_block', 'block_build')
        profile.wrap(sampler, 'annotate_outer_storage', 'storage_annotation')
        profile.wrap(sampler, '_attach_cuda_storage_rows', 'cuda_storage_rows')
        from digit import output_fusion
        profile.wrap(output_fusion, 'annotate', 'fused_storage_annotation')
    profile.wrap(loader, 'resolve_batch_feature_rows', 'feature_row_resolution')
    profile.wrap(loader, 'window_buffering', 'window_hint')
    profile.wrap(loader, '_read_many', 'merged_read')


def off_summary():
    return dict(mode='off', spans={}, added_cuda_synchronization=False,
                cuda_events=False, standalone_cuda_kernel_times=False)


def validate_profile(report):
    p = report['sampling_profile']
    if p['added_cuda_synchronization'] or p['cuda_events']:
        raise RuntimeError('Unexpected device timing instrumentation')
    if p['mode'] == 'off':
        if p['spans']:
            raise RuntimeError('Off control has active spans')
        return True
    if p['mode'] != 'host':
        raise RuntimeError('Unknown profile mode')
    n = report['updates']; spans = p['spans']
    expected = {'sampler': n, 'sampler/blocks': n}
    for layer in range(3):
        prefix = 'sampler/blocks/' + ('block_build/' if report['arm']=='digit' else '')
        expected[prefix+'layer%d.to_block' % layer] = n
        if layer or report['arm']=='gids':
            expected['sampler/blocks/layer%d.dgl_neighbors' % layer] = n
    if report['arm'] == 'digit':
        for key in ('metadata_lookup', 'output_allocation', 'native_call', 'valid_nonzero',
                    'compact_indices', 'frontier_graph', 'edge_annotations'):
            expected['sampler/blocks/outer/'+key] = n
        expected['sampler/blocks/storage_annotation'] = n
    for key, count in expected.items():
        if spans.get(key, {}).get('calls') != count:
            raise RuntimeError('Missing or partial span: '+key)
    for key, value in spans.items():
        if not 0 <= value['exclusive_seconds'] <= value['inclusive_seconds']:
            raise RuntimeError('Invalid nested timing: '+key)
    # Row resolution happens once for each hint and once for each read batch.
    resolved = sum(v['calls'] for k,v in spans.items() if k.endswith('/feature_row_resolution'))
    if resolved != 2*n:
        raise RuntimeError('Incomplete feature row resolution accounting')
    if spans['sampler']['inclusive_seconds'] > report['timing']['sample_host_seconds']:
        raise RuntimeError('Sampler span exceeds its collate envelope')
    return True
