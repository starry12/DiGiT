"""Native matched-root worker. Fresh process and caches for every arm/round."""
import argparse
import os
from pathlib import Path
from .common import *
from .sampling import graph_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--mode', choices=('smoke', 'performance'), required=True)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source = verify()
    output = args.output.resolve()
    require(output.parent == (OUT/'runs').resolve(), 'Wrong worker output root')
    require((args.mode == 'smoke' and args.round == 0) or
            (args.mode == 'performance' and 1 <= args.round <= ROUNDS), 'Wrong round')
    if not args.execute:
        print(dict(variant=args.variant, mode=args.mode, round=args.round,
                   matched_bfs_roots=True, source_sha256=source))
        return
    require(os.geteuid() == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '2',
            'Use local sudo and CUDA_VISIBLE_DEVICES=2')
    with exclusive():
        prepared = check_prepared(DEFAULT_OUTPUT)
        prepared_sha = sha(DEFAULT_OUTPUT/'prepared.json')
        gpu = read(OUT/'gpu_checks.json')
        require(gpu['passed'] and gpu['source_sha256'] == source, 'Missing GPU check')
        require(not output.exists(), 'Preserve previous output')
        if args.mode == 'performance':
            receipts = (read(p) for p in (OUT/'runs').glob('*/report.json'))
            require(any(r.get('passed') and r.get('mode') == 'smoke' and
                        r.get('matched_variant') == args.variant and
                        r.get('candidate_sha256') == source and
                        r.get('prepared_sha256') == prepared_sha for r in receipts),
                    'Missing matching smoke')
        from candidates.uks_freq_bfs_retry_v1 import worker as base
        from ae.common import check_device
        base.admission()
        check_device()
        output.mkdir(parents=True)
        arm = 'gids' if args.variant == 'gids' else 'digit'
        if args.mode == 'smoke':
            from candidates.uks_freq_bfs_retry_v1 import smoke as target
            entry = lambda: target.smoke(arm, output)
        else:
            target = base
            entry = lambda: base.run('gids_default' if arm == 'gids' else 'digit_cpu2', output, 'off')
        previous_sampler, previous_roots = target.graph_sampler, target.load_roots
        target.graph_sampler = lambda a, seed: graph_sampler(a, seed, DEFAULT_OUTPUT, args.variant == 'reuse')
        target.load_roots = matched_roots
        try:
            from candidates.uks_freq_bfs_v1.common import digest
            expected = digest(matched_roots(arm))
            result = entry()
            if args.mode == 'performance':
                require(result['roots_sha256'] == expected, 'Timed roots differ from accepted BFS window')
            require(verify() == source and check_prepared(DEFAULT_OUTPUT) == prepared,
                    'Inputs/source changed')
            result.update(passed=True, mode=args.mode, matched_variant=args.variant,
                          round=args.round, candidate_sha256=source, prepared_sha256=prepared_sha,
                          bfs_enabled=True, order_policy='full_training_induced_bfs_block_shuffle_window',
                          matched_window_roots_sha256=expected, anonymous_host_csc=True,
                          raw_ssd_writes=False, feature_layout_changed=False,
                          performance_result=args.mode == 'performance')
            write(output/'report.json', result)
        except BaseException as error:
            write(output/'failure.json', dict(passed=False, error=repr(error), candidate_sha256=source))
            raise
        finally:
            target.graph_sampler, target.load_roots = previous_sampler, previous_roots


if __name__ == '__main__':
    main()
