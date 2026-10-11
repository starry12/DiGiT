"""Small GPU-only trace verifies both captured passes appear in Kineto output."""
from .common import *
from .trace import kernel_evidence, compare_traces


def main():
    import subprocess
    require(not (OUT/'trace_probe.json').exists(), 'Preserve probe receipt')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'Use physical GPU2')
    fields = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    require(fields[0].strip() == GPU_UUID and int(fields[1]) < 1024, 'GPU2 busy or changed')
    setup()
    import torch
    from .model import DenseGraphSAGE, prepare_model
    from .equivalence import make_batch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = ('common.py', 'model.py', 'equivalence.py', 'trace.py', 'trace_probe.py')
    tested = {str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names}
    blocks, x, labels = make_batch(1024, 0)
    traces = {}
    for mode in VARIANTS:
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = DenseGraphSAGE(128,128,172,3,.2).cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001)
        prepare_model(model,optimizer,mode)
        path = OUT/('probe_'+mode+'_trace.json')
        require(not path.exists(), 'Preserve trace')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=2,warmup=1,active=16,repeat=1),
                on_trace_ready=lambda p:p.export_chrome_trace(str(path)),
                record_shapes=False,profile_memory=False,with_stack=False) as profiler:
            for _ in range(20):
                loss = torch.nn.functional.cross_entropy(model(blocks,x),labels)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                profiler.step()
        torch.cuda.synchronize()
        traces[mode] = kernel_evidence(path,mode)
        print(mode, traces[mode], flush=True)
        del model,optimizer
        torch.cuda.empty_cache()
    require(tested == {str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names}, 'Probe sources changed')
    write(OUT/'trace_probe.json',dict(passed=True,comparison=compare_traces(traces['eager'],traces['graph']),
        raw_ssd_access=False,performance_evidence=False,model_updates=40,tested_sha256=tested))


if __name__ == '__main__':main()
