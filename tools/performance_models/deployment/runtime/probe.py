"""Bounded synthetic GAT/Adam peak before any large graph or SSD ownership."""
import gc
import math
from .models import DIMENSIONS, contract, make_model, model_memory


def validate(receipt, dataset, model):
    expected = model_memory(dataset, model)['blocks']
    return (receipt.get('passed') is True and receipt.get('dataset') == dataset
            and receipt.get('model') == model and receipt.get('blocks') == expected
            and receipt.get('updates') == 4 and receipt.get('finite') is True
            and receipt.get('raw_ssd_access') is False
            and 0 < receipt.get('peak_reserved_bytes', 0) <= receipt.get('model_allowance_bytes', 0)
            <= receipt.get('probe_limit_bytes', 0)
            and receipt.get('model_allowance_bytes') == math.ceil(receipt['peak_reserved_bytes'] * 1.25) + 256*2**20)


def run(dataset, model, limit_bytes):
    import torch
    import dgl
    if not torch.cuda.is_initialized() or not 2**30 <= limit_bytes <= 12*2**30:
        raise RuntimeError('Admitted CUDA initialization and bounded probe limit required')
    from candidates.ukl_native_sampling_v10r4 import fork_guard as guard
    if guard._ACTIVE:
        raise RuntimeError('Synthetic model probe must precede graph ownership')
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    if free < limit_bytes + 2**30:
        raise RuntimeError('Insufficient free GPU memory for bounded model probe')
    previous_fraction = torch.cuda.get_per_process_memory_fraction(device) if hasattr(torch.cuda, 'get_per_process_memory_fraction') else 1.0
    # Reserve room for model already initialized by the caller. Fail before
    # OOM can spill into another GPU; no device selection occurs here.
    baseline = torch.cuda.memory_reserved(device)
    cap = (baseline + limit_bytes) / total
    torch.cuda.set_per_process_memory_fraction(min(previous_fraction, cap), device)
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    blocks = net = opt = features = targets = loss = predictions = None
    try:
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        specs = model_memory(dataset, model)['blocks']
        blocks = []
        for b in specs:
            # Distinct neighbors achieve the maximum sampler frontier size.
            u = torch.arange(b['dst'], b['src'], device=device)
            v = torch.arange(b['dst'], device=device).repeat_interleave(b['edges'] // b['dst'])
            blocks.append(dgl.create_block((u, v), num_src_nodes=b['src'], num_dst_nodes=b['dst']))
        del u, v
        torch.random.default_generator.manual_seed(0)
        torch.cuda.manual_seed(0)
        net = make_model(dataset, model, device)
        opt = torch.optim.Adam(net.parameters(), **contract(dataset, model)['optimizer'])
        features = torch.ones((specs[0]['src'], DIMENSIONS[dataset]), device=device)
        targets = torch.arange(specs[-1]['dst'], device=device) % 19
        finite = True
        for _ in range(4):
            predictions = net(blocks, features)
            loss = torch.nn.functional.cross_entropy(predictions, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            finite = finite and bool(torch.isfinite(loss).item())
        torch.cuda.synchronize(device)
        finite = finite and all(bool(torch.isfinite(p).all().item()) for p in net.parameters())
        peak = torch.cuda.max_memory_reserved(device) - baseline
        allowance = math.ceil(peak * 1.25) + 256*2**20
        result = dict(passed=finite and allowance <= limit_bytes, dataset=dataset, model=model,
                      blocks=specs, updates=4, finite=finite, raw_ssd_access=False,
                      peak_reserved_bytes=peak, model_allowance_bytes=allowance,
                      probe_limit_bytes=limit_bytes, native_acceptance=False)
        if not validate(result, dataset, model):
            raise RuntimeError('Synthetic GAT model does not fit remaining GPU budget: ' + repr(result))
        return result
    finally:
        blocks = net = opt = features = targets = loss = predictions = None
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(previous_fraction, device)
        torch.set_rng_state(state)
        torch.cuda.set_rng_state(cuda_state, device)
