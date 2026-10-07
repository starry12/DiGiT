"""Fresh-process CUDA math/Adam and maximum-block probes, without SSD access."""
import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path


def math_check(dataset, model):
    import torch
    from .models import make_model, DIMENSIONS, state_hash
    from .tests import blocks, oracle
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cpu = make_model(dataset, model).eval()
    gpu = make_model(dataset, model, 'cuda').eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(8, DIMENSIONS[dataset])
    bs = blocks()
    pred_cpu = oracle(cpu, bs, x, model)
    pred_gpu = gpu([b.to('cuda') for b in bs], x.cuda())
    torch.testing.assert_close(pred_cpu, pred_gpu.cpu(), rtol=2e-4, atol=2e-5)
    target = torch.tensor([2, 18])
    for net, prediction, labels in [(cpu, pred_cpu, target), (gpu, pred_gpu, target.cuda())]:
        torch.nn.functional.cross_entropy(prediction, labels).backward()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=3e-4, atol=2e-5)
    first = state_hash(gpu)
    opt = torch.optim.Adam(gpu.parameters(), lr=.001, weight_decay=.001)
    opt.step()
    if first == state_hash(gpu) or not all(torch.isfinite(p).all().item() for p in gpu.parameters()):
        raise RuntimeError('No finite GPU optimizer update')
    torch.cuda.synchronize()
    return dict(passed=True, independent_cpu_equations=True, forward_and_gradient_parity=True,
                finite_optimizer_update=True, parameters=sum(p.numel() for p in gpu.parameters()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('IG','UKS','UKL','CL'), required=True)
    p.add_argument('--model', choices=('gcn','gat'), required=True)
    p.add_argument('--limit-gib', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(passed=False, dataset=a.dataset, model=a.model, raw_ssd_access=False, native_acceptance=False)
    here=Path(__file__).resolve().parent
    sources={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('models.py','probe.py','cuda_checks.py','tests.py')}
    result['sources']=sources
    try:
        import torch
        torch.set_num_threads(1)
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8,9):
            raise RuntimeError('One selected L40 required')
        result['math'] = math_check(a.dataset, a.model)
        gc.collect()
        torch.cuda.empty_cache()
        from .probe import run
        result['maximum_blocks'] = run(a.dataset, a.model, int(a.limit_gib * 2**30))
        if any(hashlib.sha256((here/n).read_bytes()).hexdigest()!=digest for n,digest in sources.items()):
            raise RuntimeError('CUDA model-check source changed during execution')
        result['passed'] = True
    except BaseException as error:
        result.update(error=repr(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
