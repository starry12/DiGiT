"""Observe node-local capacity; enforce actual placement and host runtime limits."""
import json
import os
import time
from pathlib import Path

PAGE = 4096
CHUNK = 16 * 2**20
RESERVE = 4 * 2**30  # Per physical node; additional to the original host reserve.
VERSION = 'numa_admission_v4'
PLACEMENT_POLICY = 'stage_only'
ADMISSION_POLICY = 'node_capacity_advisory_host_and_placement_enforced'
CURRENT = None
ADAPTER_SHA = None


class NumaFailure(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('NUMA ' + json.dumps(evidence, sort_keys=True))


def node_memory():
    result = {}
    for node in (0, 1):
        fields = {}
        for line in Path('/sys/devices/system/node/node%d/meminfo' % node).read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[3].isdigit() and parts[4] == 'kB':
                fields[parts[2].rstrip(':')] = int(parts[3]) * 1024
        if not {'MemFree', 'MemTotal'} <= fields.keys():
            raise RuntimeError('Missing node-local memory counters: ' + str(node))
        result[str(node)] = fields
    return result


def capacity_observation(memory, needed):
    free, reclaimable, available = {}, {}, {}
    for node in ('0', '1'):
        v = memory[node]
        for key in ('MemFree', 'MemTotal', 'FilePages', 'Shmem', 'Dirty', 'Writeback', 'Mapped'):
            if type(v[key]) is not int or v[key] < 0:
                raise ValueError('Invalid node memory counter: ' + node + ':' + key)
        if v['MemFree'] > v['MemTotal']:
            raise ValueError('Node free memory exceeds total')
        free[node] = v['MemFree']
        reclaimable[node] = max(0, v['FilePages'] - v['Shmem'] - v['Dirty'] - v['Writeback'] - v['Mapped'])
        available[node] = min(v['MemTotal'], free[node] + reclaimable[node])
    return dict(free_bytes=free, reclaimable_file_bytes_estimate=reclaimable,
                available_bytes_estimate=available,
                free_threshold_met=all(v >= needed for v in free.values()),
                estimated_threshold_met=all(v >= needed for v in available.values()))


class Guard:
    def __init__(self, output, event, memory=node_memory, checkpoint=lambda: None):
        self.output = Path(output)
        self.event = event
        self.memory = memory
        self.checkpoint = checkpoint
        self.reclaim_attempted = set()
        self.reclaims = []
        self.admissions = {}
        self.initial_admission = {}
        self.stages = {}

    def save(self, row, failed=False):
        row = dict(row, version=VERSION, time=time.time(), pid=os.getpid())
        # Independent of status.json, which cleanup events overwrite.
        with (self.output / 'numa_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
        if failed:
            target = self.output / 'numa_failure.json'
            if not target.exists():
                tmp = target.with_suffix('.tmp')
                with tmp.open('w') as stream:
                    json.dump(row, stream, indent=2)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                os.replace(str(tmp), str(target))
            raise NumaFailure(row)
        self.event('numa_check', evidence=row)
        return row

    def admit(self, label, stage, remaining, sources=None):
        if type(remaining) is not int or remaining < 0 or remaining % PAGE:
            raise ValueError('Page-aligned remaining NUMA allocation required')
        needed = ((remaining // PAGE + 1) // 2) * PAGE + RESERVE
        row = dict(kind='admission', label=label, stage=stage,
                   remaining_bytes=remaining, per_node_reserve_bytes=RESERVE,
                   required_bytes={'0': needed, '1': needed},
                   admission_policy=ADMISSION_POLICY, free_threshold_enforced=False,
                   node_sample_pages={}, sampled_pages=0)
        try:
            self.checkpoint()
            row['node_memory'] = self.memory()
            row.update(capacity_observation(row['node_memory'], needed))
            row['passed'] = True
            if not row['free_threshold_met']:
                row['observation'] = 'free_below_full_allocation_estimate'
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            row.update(passed=False, reason='invalid_node_memory_observation', error=repr(error))
        self.save(row, failed=not row['passed'])
        self.admissions[label] = row
        if stage == 'before_first_touch':
            self.initial_admission[label] = row
        return row

    def placement(self, row, passed, reason=None):
        record = dict(row, kind='placement', passed=passed)
        if reason:
            record['reason'] = reason
        self.save(record, failed=not passed)
        if row['stage'] in ('loaded', 'cache_allocated', 'cache_preloaded', 'before_training'):
            self.stages[row['label'] + ':' + row['stage']] = row

    def receipt(self):
        return dict(version=VERSION, admission_policy=ADMISSION_POLICY, adapter_sha256=ADAPTER_SHA, passed=True, per_node_reserve_bytes=RESERVE,
                    admission=self.admissions, initial_admission=self.initial_admission, stages=self.stages,
                    placement_policy=PLACEMENT_POLICY, cache_chunks_checked=0, cache_bytes_checked=0,
                    reclaim_attempts=self.reclaims)


def valid(receipt):
    try:
        if (receipt['version'] != VERSION or receipt.get('admission_policy') != ADMISSION_POLICY or receipt['passed'] is not True
                or receipt['per_node_reserve_bytes'] != RESERVE):
            return False
        labels = {'graph', 'auxiliary', 'cpu_features'}
        if set(receipt['admission']) != labels or set(receipt['initial_admission']) != labels:
            return False
        for label, row in list(receipt['admission'].items()) + list(receipt['initial_admission'].items()):
            remaining = row['remaining_bytes']
            if type(remaining) is not int or remaining < 0 or remaining % PAGE:
                return False
            needed = ((remaining // PAGE + 1) // 2) * PAGE + RESERVE
            if (row['label'] != label or row['passed'] is not True
                    or row['per_node_reserve_bytes'] != RESERVE
                    or row['required_bytes'] != {'0': needed, '1': needed}
                    or row.get('admission_policy') != ADMISSION_POLICY
                    or row.get('free_threshold_enforced') is not False):
                return False
            expected = capacity_observation(row['node_memory'], needed)
            if any(row.get(k) != v for k, v in expected.items()):return False
        required = {'graph:loaded', 'auxiliary:loaded', 'cpu_features:cache_allocated',
                    'cpu_features:cache_preloaded'} | {n + ':before_training' for n in labels}
        if set(receipt['stages']) != required:
            return False
        sizes = {}
        for key, row in receipt['stages'].items():
            if key != row['label'] + ':' + row['stage'] or row['verified'] is not True:
                return False
            size = row['bytes']; counts = row['node_sample_pages']; total = row['sampled_pages']
            if (type(size) is not int or size < 2 * PAGE or size % PAGE
                    or row['nodes'] != [0, 1] or row['policy'] != 'interleave'
                    or row['migration'] is not False or row['before_first_touch'] is not True
                    or row['offset_bytes'] != 0
                    or row['verification'] != 'range_policy_and_stratified_move_pages_query'
                    or type(total) is not int or total != min(4096, size // PAGE // 2 * 2)
                    or set(counts) != {'0', '1'}
                    or any(type(v) is not int or not 2 * total <= 5 * v <= 3 * total for v in counts.values())
                    or sum(counts.values()) != total):
                return False
            if row['label'] in sizes and sizes[row['label']] != size:
                return False
            sizes[row['label']] = size
        if any(receipt['initial_admission'][label]['stage'] != 'before_first_touch'
               or receipt['initial_admission'][label]['remaining_bytes'] != sizes[label] for label in labels):
            return False
        return (receipt['placement_policy'] == PLACEMENT_POLICY
                and type(receipt['cache_chunks_checked']) is int and receipt['cache_chunks_checked'] == 0
                and type(receipt['cache_bytes_checked']) is int and receipt['cache_bytes_checked'] == 0)
    except (KeyError, TypeError, ValueError):
        return False
