"""Bounded GPU-only API/ordering tests. No controller, graph or SSD is opened."""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from .native_sources import BINARY, HERE, native_sources, object_sha, sha


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(fn):
    try:
        fn()
    except (RuntimeError, ValueError):
        return True
    raise AssertionError('Invalid async operation was accepted')


def run():
    import torch
    torch.set_num_threads(1)
    check(torch.cuda.device_count() == 1, 'Expose only the selected GPU for this fixture')
    sys.path.insert(0, str(HERE / 'runtime'))
    import BAM_Feature_Store as native
    check(native.ASYNC_FEATURE_READ_API == 1, 'Wrong native async API')
    check(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve() == BINARY,
          'Wrong native binary')
    build = json.loads((HERE / 'runtime/build_receipt.json').read_text())
    check(sha(BINARY) == build['binary_sha256'], 'Native binary changed')
    check(object_sha(native_sources()) == build['native_source_sha256'], 'Native sources changed')
    io = torch.cuda.Stream()
    model = torch.cuda.Stream()
    probe = native.AsyncFeatureProbe(io.cuda_stream)
    hint = torch.zeros(1, dtype=torch.int64, device='cuda')
    ids = [torch.tensor([0, 1, 3, 9, 2], device='cuda', dtype=torch.int64),
           torch.tensor([4, 3, 0], device='cuda', dtype=torch.int64),
           torch.tensor([2, 11, 9, 12], device='cuda', dtype=torch.int64),
           torch.tensor([0], device='cuda', dtype=torch.int64)]
    outputs = [torch.full((len(x), 128), float('nan'), device='cuda') for x in ids]
    expected_cpu = sum(int((x.cpu() % 3 == 0).sum()) for x in ids)
    arguments = ([x.data_ptr() for x in outputs], [x.data_ptr() for x in ids], [len(x) for x in ids])
    io.wait_stream(torch.cuda.current_stream())
    failures_checked = []

    def invalid(name, fn):
        rejects(fn)
        failures_checked.append(name)

    invalid('default_stream_rejected', lambda: native.AsyncFeatureProbe(0))
    invalid('uninitialized_real_store_rejected',
            lambda: native.BAM_Feature_Store_float().prepare_async_reads(io.cuda_stream))
    invalid('too_many_batches', lambda: probe.enqueue(arguments[0] + arguments[0][:1],
            arguments[1] + arguments[1][:1], arguments[2] + arguments[2][:1],
            hint.data_ptr(), 0, io.cuda_stream))
    invalid('empty_batches', lambda: probe.enqueue([], [], [], hint.data_ptr(), 0, io.cuda_stream))
    invalid('mismatched_vectors', lambda: probe.enqueue(arguments[0], [], arguments[2],
            hint.data_ptr(), 0, io.cuda_stream))
    invalid('wrong_io_stream', lambda: probe.hint(hint.data_ptr(), 7, model.cuda_stream))

    probe.hint(hint.data_ptr(), 7, io.cuda_stream)
    ticket = probe.enqueue(*arguments, hint.data_ptr(), 0, io.cuda_stream)
    invalid('reuse_before_collect', lambda: probe.enqueue(*arguments, hint.data_ptr(), 0, io.cuda_stream))
    invalid('wrong_ticket', lambda: probe.collect(ticket + 1))
    invalid('pending_idle_guard', probe.require_idle)
    invalid('drain_before_collect', probe.drain)
    # Hint follows this request's complete marker on the same IO stream.
    probe.hint(hint.data_ptr(), 23, io.cuda_stream)
    count = probe.collect(ticket)
    check(count == expected_cpu, 'First request CPU count differs')
    invalid('duplicate_collect', lambda: probe.collect(ticket))
    invalid('query_collected_ticket', lambda: probe.query(ticket))
    invalid('wait_collected_ticket', lambda: probe.wait(ticket, model.cuda_stream))
    for idx, output in zip(ids, outputs):
        expected = idx[:, None] * 128 + torch.arange(128, device='cuda')[None, :] + 7
        check(torch.equal(output, expected.float()), 'First request hint/data ordering differs')

    ticket = probe.enqueue(*arguments, hint.data_ptr(), 0, io.cuda_stream)
    probe.wait(ticket, model.cuda_stream)
    with torch.cuda.stream(model):
        consumed = [x + 1 for x in outputs]
    check(probe.collect(ticket) == expected_cpu, 'Second request counter was not independent')
    model.synchronize()
    for idx, output in zip(ids, consumed):
        expected = idx[:, None] * 128 + torch.arange(128, device='cuda')[None, :] + 24
        check(torch.equal(output, expected.float()), 'Consumer event dependency differs')
    check(probe.cpu_total == expected_cpu * 2 and probe.requests == 2, 'Counter total double-counted')
    thread_result = []

    def wrong_thread():
        try:
            probe.require_idle()
        except RuntimeError:
            thread_result.append(True)
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    check(thread_result == [True], 'Second host owner was accepted')
    failures_checked.append('second_host_owner')

    # Warm lazy kernel loading and allocator work before measuring event spans.
    native.async_probe_busy_wait(model.cuda_stream, 0)
    model.synchronize()
    probe.drain()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    read_end = torch.cuda.Event(enable_timing=True)
    model_start = torch.cuda.Event(enable_timing=True)
    model_end = torch.cuda.Event(enable_timing=True)
    start.record(io)
    begin = time.perf_counter()
    ticket = probe.enqueue(*arguments, hint.data_ptr(), 30000000, io.cuda_stream)
    enqueue_host_seconds = time.perf_counter() - begin
    read_end.record(io)
    model.wait_event(start)
    model_start.record(model)
    native.async_probe_busy_wait(model.cuda_stream, 300000000)
    model_end.record(model)
    check(probe.collect(ticket) == expected_cpu, 'Delayed request counter differs')
    independent_work_still_running = not model_end.query()
    model_end.synchronize()
    read_end.synchronize()
    read_ms = start.elapsed_time(read_end)
    model_start_ms = start.elapsed_time(model_start)
    model_end_ms = start.elapsed_time(model_end)
    overlap_ms = max(0.0, min(read_ms, model_end_ms) - max(0.0, model_start_ms))
    check(independent_work_still_running, 'Collect unexpectedly waited for independent GPU work')
    check(overlap_ms > 1.0, 'No event-confirmed stream overlap observed')
    for idx, output in zip(ids, outputs):
        expected = idx[:, None] * 128 + torch.arange(128, device='cuda')[None, :] + 23
        check(torch.equal(output, expected.float()), 'Delayed request produced wrong features')
    check(probe.cpu_total == expected_cpu * 3 and probe.requests == 3, 'Delayed counter total differs')
    probe.drain()

    # Destruction with a pending request drains only owned operations. Retain
    # pointer owners and the borrowed IO stream until the native state is gone.
    lifetime = native.AsyncFeatureProbe(io.cuda_stream)
    lifetime.enqueue(*arguments, hint.data_ptr(), 1000000, io.cuda_stream)
    del lifetime
    for idx, output in zip(ids, outputs):
        expected = idx[:, None] * 128 + torch.arange(128, device='cuda')[None, :] + 23
        check(torch.equal(output, expected.float()), 'Destruction freed a live request')
    return dict(passed=True, gpu_only=True, raw_ssd_reads=False, raw_ssd_writes=False,
                binary_sha256=sha(BINARY), native_source_sha256=build['native_source_sha256'],
                probe_python_sha256=sha(Path(__file__)), device=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                merged_batches=4, feature_rows=sum(arguments[2]), cpu_count_per_request=expected_cpu,
                request_counts_exact=True, output_elements_exact=True,
                async_hint_and_consumer_event_ordering_exact=True,
                rejected_operations=failures_checked, pending_destructor_safe=True,
                synthetic_overlap=dict(read_interval_ms=[0.0, read_ms],
                    model_interval_ms=[model_start_ms, model_end_ms], overlap_ms=overlap_ms,
                    enqueue_host_seconds=enqueue_host_seconds,
                    collect_did_not_wait_for_independent_work=independent_work_still_running),
                limitations=['Synthetic gather uses production async scheduling, no BaM cache or SSD',
                             'This proves event/count/lifetime behavior, not real training speedup'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'runtime/native_probe.json')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous native probe evidence')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
