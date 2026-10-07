"""Explicit process-local model substitution around immutable sampler/I/O code.

Parent source hashes still describe the parent implementation. Every new report
also requires a separate adapter hash and model contract. Never patch disk files.
"""
import copy
import importlib
from .models import contract, make_model, model_memory, require_identity, LARGE_GAT_MODEL_LIMIT

_ACTIVE = None


def activate(dataset, model, expected):
    global _ACTIVE
    if _ACTIVE is not None:
        if _ACTIVE != expected:
            raise RuntimeError('Cannot switch model contracts inside one process')
        return
    if expected['dataset'] != dataset or expected['model'] != model:
        raise RuntimeError('Activation identity mismatch')
    contract(dataset, model)
    if dataset == 'IG':
        module = importlib.import_module('candidates.ig_sage_host_telemetry_v1.model')
        module.make_model = lambda name, device='cpu', dtype=None: make_model(dataset, _same(name, model), device, dtype)
        validation = importlib.import_module('candidates.ig_sage_host_telemetry_v1.validation')
        original = validation.report_check
        def check(report, *args, **kwargs):
            require_identity(report, expected)
            from .gates import validate as gate_valid
            if not gate_valid(report.get('warmup_gate')):
                raise RuntimeError('Missing IG new-model warmup gate')
            if report.get('model_name') != model:
                raise RuntimeError('Worker model mismatch')
            return original(report, *args, **kwargs)
        validation.report_check = check
    elif dataset == 'UKS':
        module = importlib.import_module('candidates.uks_native_v1.model')
        def create(p, device='cpu'):
            import torch
            if device != 'cpu':
                module.heavy_gate()
            else:
                module.require(p.get('fixture') and p['nodes'] <= 4096, 'CPU fixture required')
            c = contract(dataset, model)
            for key, value in [('hidden', 128), ('classes', 19), ('layers', 3), ('dropout', .2), ('seed', 0)]:
                module.require(p[key] == value, 'Inherited model configuration changed: ' + key)
            module.require({k: p['optimizer'][k] for k in c['optimizer']} == c['optimizer'], 'Optimizer changed')
            from candidates.pa_sage_cache_policy_v1.training import seed_cpu, model_hash
            seed_cpu(p['seed'])
            net = make_model(dataset, model, device)
            opt = torch.optim.Adam(net.parameters(), **c['optimizer'])
            initial = model_hash(net)
            seed_cpu(p['seed'])
            net.train()
            return net, opt, initial
        module.create = create
    else:
        _large_graph(dataset, model, expected)
    _ACTIVE = expected


def _same(actual, expected):
    if actual != expected:
        raise ValueError('Unexpected model in fixed service')
    return actual


def _large_graph(dataset, model, expected):
    package = 'candidates.' + {'UKL': 'ukl_common_gpu_five_v27', 'CL': 'cl_common_gpu_five_v17'}[dataset]
    budget_module = importlib.import_module(package + '.budget')
    original_budget = budget_module.budget
    model_probe = {}
    def budget(*args, **kwargs):
        result = copy.deepcopy(original_budget(*args, **kwargs))
        inherited_margin = result['gpu_free_min'] - result['gpu_accounted']
        if inherited_margin < 0:
            raise RuntimeError('Invalid inherited GPU budget')
        parts = result['gpu_components']
        allowance = model_probe.get('model_allowance_bytes', model_memory(dataset, model)['bytes'])
        minimum = LARGE_GAT_MODEL_LIMIT if model == 'gat' else parts['model_blocks_activations']
        parts['model_blocks_activations'] = max(minimum, allowance)
        result['gpu_accounted'] = sum(parts.values())
        result['gpu_free_min'] = max(result['gpu_free_min'], result['gpu_accounted'] + inherited_margin)
        result['model_identity'] = expected
        return result
    budget_module.budget = budget
    for name, attr in [('protocol', 'training_budget'), ('backend', 'budget'), ('admission', 'budget')]:
        setattr(importlib.import_module(package + '.' + name), attr, budget)
    training = importlib.import_module(package + '.training')
    def prepare_model(seed=0):
        import torch
        from candidates.ukl_native_sampling_v10r4 import fork_guard as guard
        if guard._ACTIVE or torch.cuda.is_initialized():
            raise RuntimeError('CPU preparation must precede graph ownership and CUDA')
        guard.prewarm_cpu_block()
        state = torch.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(seed)
            scratch = make_model(dataset, model)
            opt = torch.optim.Adam(scratch.parameters(), **contract(dataset, model)['optimizer'])
            sum(p.square().sum() for p in scratch.parameters()).backward()
            opt.step()
            del scratch, opt
            torch.random.default_generator.manual_seed(seed)
            net = make_model(dataset, model)
            return net, torch.optim.Adam(net.parameters(), **contract(dataset, model)['optimizer'])
        finally:
            torch.set_rng_state(state)
    training.prepare_model = prepare_model
    from .gates import WarmupGate
    original_init = training.Engine.__init__
    def engine_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.model_gate = WarmupGate(self.model, self.opt)
    training.Engine.__init__ = engine_init
    performance = importlib.import_module(package + '.performance_engine')
    original_step = performance.Engine.step
    def step(self, *args, **kwargs):
        if self.updates in (4, 20):
            self.model_gate.require()
        loss, count = original_step(self, *args, **kwargs)
        if self.updates <= 4:
            checks = sum(v.get('passed') is True for v in self.feature_checks)
            self.model_gate.observe(loss, self.updates, checks)
        return loss, count
    performance.Engine.step = step
    if model == 'gat':
        original_gpu = training.prepare_gpu_model
        def prepare_gpu_model(seed=0):
            model_and_optimizer = original_gpu(seed)
            from .probe import run
            protocol = importlib.import_module(package + '.protocol')
            parent = original_budget(protocol.arm())
            parts = parent['gpu_components']
            # Preserve each arm's original metadata/cache allocation and its
            # existing headroom (CL GIDS has less headroom than UKL GIDS).
            limit = LARGE_GAT_MODEL_LIMIT
            model_probe.update(run(dataset, model, limit))
            return model_and_optimizer
        training.prepare_gpu_model = prepare_gpu_model
        gpu_budget = importlib.import_module(package + '.gpu_budget')
        original_admit = gpu_budget.admit_initialized
        def admit_initialized(initial_budget, free_bytes, total_bytes):
            from .probe import validate as probe_valid
            if not probe_valid(model_probe, dataset, model):
                raise RuntimeError('GAT maximum-block probe required before graph load')
            protocol = importlib.import_module(package + '.protocol')
            initial_budget.update(budget(protocol.arm()))
            return original_admit(initial_budget, free_bytes, total_bytes)
        gpu_budget.admit_initialized = admit_initialized
    original_window = training.Engine.window
    def window(self, *args, **kwargs):
        result = original_window(self, *args, **kwargs)
        result['model_identity'] = expected
        result['warmup_gate'] = self.model_gate.require()
        if model == 'gat':
            result['model_memory_probe'] = dict(model_probe)
        return result
    training.Engine.window = window
    protocol = importlib.import_module(package + '.protocol')
    original_validate = protocol.validate_worker_report
    def validate(report, *args, **kwargs):
        try:
            require_identity(report, expected)
            from .gates import validate as gate_valid
            if not gate_valid(report.get('warmup_gate')):
                return False
            if model == 'gat':
                from .probe import validate as probe_valid
                if not probe_valid(report.get('model_memory_probe', {}), dataset, model):
                    return False
        except RuntimeError:
            return False
        return original_validate(report, *args, **kwargs)
    protocol.validate_worker_report = validate
