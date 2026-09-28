"""Host spans only: preserve every original operation and synchronization site.

The off worker executes its original function. The host worker adds only with
scopes through a checked AST transform; stripping those scopes must reproduce
the original AST exactly. No events, device calls, or per-batch file writes.
"""
import ast
import hashlib
import importlib
import inspect
import math
import textwrap
import time
from contextlib import contextmanager
from functools import wraps


TOP = ('roots_transfer', 'sampling', 'feature_fetch', 'labels_transfer',
       'forward', 'loss', 'zero_grad', 'backward', 'adam', 'bookkeeping',
       'window_drain')


def dump(node):
    return ast.dump(node, include_attributes=False)


WORKER_RULES = {
    'roots_transfer': "rows=np.asarray(roots[i*p['batch_size']:(i+1)*p['batch_size']],dtype=np.int64);root=torch.from_numpy(rows.copy()).cuda()",
    'sampling': 'inputs,outputs,blocks=phase_sampler.sample_blocks(graph,root)',
    'feature_fetch': 'x=fetch(inputs,blocks,False)',
    'labels_transfer': "targets=np.asarray(labels[rows]);require(np.isfinite(targets).all() and np.all(targets==targets.astype('int64')) and targets.min()>=0 and targets.max()<19,'Invalid supervised labels');y=torch.from_numpy(targets.astype('int64')).cuda()",
    'forward': 'pred=model(blocks,x)',
    'loss': 'loss=torch.nn.functional.cross_entropy(pred,y)',
    'zero_grad': 'opt.zero_grad(set_to_none=True)',
    'backward': 'loss.backward()',
    'adam': 'opt.step()',
    'bookkeeping': "pending.append(loss.detach());examples+=len(rows);root_hash.update(rows.tobytes());window_root_hash.update(rows.tobytes());rows_seen+=len(inputs);window_rows+=len(inputs);shape_totals+=np.array([[b.num_src_nodes(),b.num_dst_nodes(),b.num_edges()] for b in blocks]);outer_edges+=blocks[0].num_edges()\nif arrays is not None:pending_groups.append(blocks[0].dstdata[DIGIT_SAMPLED_GROUPS].sum())",
    'address_mapping_d2h': "logical=inputs.cpu().numpy();physical=np.asarray(arrays['node_to_primary_row'][logical],dtype=np.int64) if arrays is not None and standard else blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy() if arrays is not None else logical;flags=blocks[0].srcdata[DIGIT_STORAGE_IS_GROUP].cpu().numpy() if arrays is not None and not standard else np.zeros(len(logical),dtype=np.bool_)",
    'feature_allocation': "x=torch.empty((len(logical),1024),device='cuda')",
    'chunk_fetch_copy': 'x[at:at+16384].copy_(features.fetch(physical[at:at+16384],flags[at:at+16384]))',
}

FEATURE_RULES = {
    'request_validation': "rows=np.ascontiguousarray(rows,dtype=np.int64);flags=np.ascontiguousarray(flags,dtype=np.bool_);require(rows.ndim==1 and flags.shape==rows.shape and 0<len(rows)<=16384 and np.all((rows>=0)&(rows<self.plan['rows'])),'Native request bounds')\nif self.arm=='gids':require(not flags.any(),'GIDS group request')\nelif flags.any():require(np.all(rows[flags]<self.hot_start),'Group request includes hot region')",
    'request_h2d_allocation': "ids=torch.from_numpy(rows).cuda();groups=torch.from_numpy(flags).cuda();out=torch.empty((len(rows),1024),device='cuda')",
    'native_read': 'began=time.perf_counter();self.store.read_feature(out.data_ptr(),ids.data_ptr(),len(rows),1024,1024,0,groups.data_ptr());self.feature_seconds+=time.perf_counter()-began',
}


class StripSpans(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if (len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call)
                and dump(node.items[0].context_expr.func) == dump(ast.parse('PROFILE.span', mode='eval').body)):
            return node.body
        return node


def transform(tree, kind):
    rules = WORKER_RULES if kind == 'worker' else FEATURE_RULES
    patterns = [(name, [dump(n) for n in ast.parse(code).body]) for name, code in rules.items()]
    counts = {name: 0 for name in rules}
    if kind == 'worker':
        counts['window_drain'] = 0

    class AddSpans(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field, values in ast.iter_fields(node):
                if not isinstance(values, list) or not values or not all(isinstance(x, ast.stmt) for x in values):
                    continue
                result = []
                i = 0
                while i < len(values):
                    label = None
                    size = 0
                    for name, pattern in patterns:
                        if [dump(x) for x in values[i:i+len(pattern)]] == pattern:
                            label, size = name, len(pattern)
                            break
                    if (kind == 'worker' and i+1 < len(values)
                            and dump(values[i]) == dump(ast.parse('torch.cuda.synchronize()').body[0])
                            and dump(values[i+1]) == dump(ast.parse('window_seconds=time.perf_counter()-window_started').body[0])):
                        label, size = 'window_drain', 1
                    if label is None:
                        result.append(values[i]); i += 1
                    else:
                        scope = ast.With(items=[ast.withitem(context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id='PROFILE', ctx=ast.Load()), attr='span', ctx=ast.Load()),
                            args=[ast.Constant(value=label)], keywords=[]), optional_vars=None)],
                            body=values[i:i+size], type_comment=None)
                        result.append(ast.copy_location(scope, values[i]))
                        counts[label] += 1
                        i += size
                setattr(node, field, result)
            return node

    original = dump(tree)
    tree = ast.fix_missing_locations(AddSpans().visit(tree))
    import copy
    if dump(StripSpans().visit(copy.deepcopy(tree))) != original:
        raise RuntimeError('Instrumentation changed executable AST')
    if any(count != 1 for count in counts.values()):
        raise RuntimeError('Missing/duplicated instrumentation sites: ' + repr(counts))
    return tree, counts


def instrument_function(function, kind, profile):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    tree, sites = transform(tree, kind)
    namespace = dict(function.__globals__, PROFILE=profile)
    exec(compile(tree, inspect.getsourcefile(function) + ':host-profile', 'exec'), namespace)
    profile.instrumentation.append(dict(function=function.__qualname__, sites=sites,
        original_ast_sha256=hashlib.sha256(dump(ast.parse(textwrap.dedent(inspect.getsource(function)))).encode()).hexdigest(),
        profiled_ast_sha256=hashlib.sha256(dump(tree).encode()).hexdigest(), stripped_ast_equal=True))
    return namespace[function.__name__]


class HostProfile:
    def __init__(self, mode, clock=time.perf_counter_ns):
        if mode not in ('off', 'host'):
            raise ValueError('Unknown profiling mode')
        self.mode, self.clock = mode, clock
        self.active = False
        self.stack, self.windows, self.restores, self.instrumentation = [], [], [], []
        self.records = {}

    @contextmanager
    def span(self, name):
        if not self.active or self.mode == 'off':
            yield
            return
        start = self.clock()
        frame = [name, 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = self.clock() - start
            assert self.stack.pop() is frame
            path = '/'.join([f[0] for f in self.stack] + [name])
            self.records.setdefault(path, []).append((elapsed, elapsed-frame[1]))
            if self.stack:
                self.stack[-1][1] += elapsed

    def start_window(self):
        if self.stack:
            raise RuntimeError('Previous window has open spans')
        self.records = {}
        self.active = self.mode == 'host'

    def finish_window(self, phase, record):
        self.active = False
        if self.stack:
            raise RuntimeError('Cannot summarize incomplete spans')
        spans = {}
        for name, values in self.records.items():
            ordered = sorted(a for a, _ in values)
            spans[name] = dict(calls=len(values), inclusive_seconds=sum(ordered)/1e9,
                exclusive_seconds=sum(b for _, b in values)/1e9,
                p50_us=ordered[(len(ordered)-1)//2]/1000,
                p95_us=ordered[math.ceil(.95*len(ordered))-1]/1000,
                max_us=ordered[-1]/1000)
        accounted = sum(v['inclusive_seconds'] for k, v in spans.items() if '/' not in k)
        self.windows.append(dict(phase=phase, index=record['index'], batches=record['batches'],
            e2e_seconds=record['seconds'], spans=spans, top_level_seconds=accounted,
            uninstrumented_seconds=record['seconds']-accounted if self.mode == 'host' else None))

    def replace(self, obj, name, value):
        local = name in vars(obj)
        self.restores.append((obj, name, vars(obj)[name] if local else None, local))
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
            raise RuntimeError('Unclosed spans')
        for obj, name, original, local in reversed(self.restores):
            if local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.restores.clear()

    def result(self):
        return dict(mode=self.mode, windows=self.windows, instrumentation=self.instrumentation,
            added_cuda_synchronization=False, cuda_events=False,
            timing='Host wall spans, including waits at existing sites; not isolated GPU kernel times.',
            nesting='Nested spans are included in parents. Never add them to top-level spans.',
            mapping='Address annotation inside sampling and mapping/copies inside feature_fetch are reported separately.',
            off_control='Fresh process with unchanged original execute/fetch functions and no timing wrappers.')


def install(profile, sampler, graph, features):
    if profile.mode == 'off':
        return
    import dgl
    from digit.sampler import DiGiTNeighborSampler
    profile.wrap(graph, 'sample_neighbors', 'dgl_neighbors')
    # GIDS imports to_block by name; DiGiT resolves it through the dgl module.
    target = dgl if isinstance(sampler, DiGiTNeighborSampler) else importlib.import_module(type(sampler).__module__)
    profile.wrap(target, 'to_block', 'to_block')
    if isinstance(sampler, DiGiTNeighborSampler):
        for name, label in [('sample_outer_frontier', 'outer_frontier'),
                            ('build_block', 'block_build'),
                            ('annotate_outer_storage', 'storage_annotation')]:
            profile.wrap(sampler, name, label)
        from uva_sampler import UVAMetadata, PinnedBuffer
        profile.wrap(UVAMetadata, 'sample', 'uva_sample_existing_waits')
        profile.wrap(PinnedBuffer, '__getitem__', 'uva_address_gather_existing_waits')
    profile.replace(type(features), 'fetch', instrument_function(type(features).fetch, 'features', profile))


def validate(r):
    p = r['stage_profile']
    if p['mode'] not in ('off', 'host') or p['added_cuda_synchronization'] or p['cuda_events']:
        raise RuntimeError('Unexpected timing mode')
    expected = [(name, w) for name in ('warmup', 'training') if r[name]
                for w in r[name]['windows']]
    if len(p['windows']) != len(expected):
        raise RuntimeError('Incomplete profile windows')
    for actual, (phase, window) in zip(p['windows'], expected):
        if (actual['phase'], actual['index'], actual['batches'], actual['e2e_seconds']) != (phase, window['index'], window['batches'], window['seconds']):
            raise RuntimeError('Profile/E2E window differs')
        spans = actual['spans']
        if p['mode'] == 'off':
            if spans or p['instrumentation'] or actual['top_level_seconds'] != 0:
                raise RuntimeError('Off worker has instrumentation')
            continue
        for key in TOP:
            count = 1 if key == 'window_drain' else window['batches']
            if spans.get(key, {}).get('calls') != count:
                raise RuntimeError('Missing/partial stage: ' + key)
        for key in ('address_mapping_d2h', 'feature_allocation'):
            if spans.get('feature_fetch/'+key, {}).get('calls') != window['batches']:
                raise RuntimeError('Incomplete mapping measurement')
        prefix = 'feature_fetch/chunk_fetch_copy'
        chunks = spans.get(prefix, {}).get('calls', 0)
        if chunks < window['batches'] or any(spans.get(prefix+'/'+key, {}).get('calls') != chunks for key in FEATURE_RULES):
            raise RuntimeError('Incomplete feature chunk instrumentation')
        block = 'sampling/block_build/to_block' if r['arm'] == 'digit_full' else 'sampling/to_block'
        if spans.get(block, {}).get('calls') != 3*window['batches']:
            raise RuntimeError('Incomplete block measurement')
        if spans.get('sampling/dgl_neighbors', {}).get('calls') != (2 if r['arm']=='digit_full' else 3)*window['batches']:
            raise RuntimeError('Incomplete DGL sampling measurement')
        if r['arm'] == 'digit_full':
            for key in ('outer_frontier/uva_sample_existing_waits', 'storage_annotation', 'storage_annotation/uva_address_gather_existing_waits'):
                if spans.get('sampling/'+key, {}).get('calls') != window['batches']:
                    raise RuntimeError('Incomplete DiGiT measurement: '+key)
        for key, span in spans.items():
            if not (math.isfinite(span['inclusive_seconds']) and 0 <= span['exclusive_seconds'] <= span['inclusive_seconds']):
                raise RuntimeError('Invalid time: '+key)
        accounted = sum(v['inclusive_seconds'] for k,v in spans.items() if '/' not in k)
        if not math.isclose(accounted, actual['top_level_seconds'], abs_tol=1e-9) or accounted > window['seconds']:
            raise RuntimeError('Stages exceed synchronized E2E')
        if not math.isclose(accounted+actual['uninstrumented_seconds'], window['seconds'], abs_tol=1e-9):
            raise RuntimeError('Unreconciled E2E')
    return True
