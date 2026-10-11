"""Strict node-local admission with bounded source-cache recovery; no migration."""
import json
import os
import time
from pathlib import Path

PAGE = 4096
CHUNK = 16 * 2**20
RESERVE = 4 * 2**30  # Per physical node; additional to the original host reserve.
VERSION = 'numa_admission_v2'
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
        self.chunk_count = 0
        self.chunk_bytes = 0

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
                   node_sample_pages={}, sampled_pages=0)
        try:
            row['node_memory'] = self.memory()
            row['free_bytes'] = {n: row['node_memory'][n]['MemFree'] for n in ('0', '1')}
            row['passed'] = all(v >= needed for v in row['free_bytes'].values())
            if not row['passed'] and sources and label not in self.reclaim_attempted:
                self.reclaim_attempted.add(label)
                import importlib.util
                spec = importlib.util.spec_from_file_location("_numa_reclaim", Path(__file__).with_name("reclaim.py"))
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                recover = module.recover
                # Save the shortage even if recovery itself fails or is interrupted.
                self.save(dict(row, kind='admission_shortage_before_reclaim'))
                recovery = recover(sources, needed, self.memory, self.checkpoint,
                                   lambda evidence: self.save(dict(evidence, label=label, stage=stage)))
                self.reclaims.append(recovery)
                row['reclaim'] = recovery
                # Re-read, never turn an advisory success into memory acceptance.
                row['node_memory'] = self.memory()
                row['free_bytes'] = {n: row['node_memory'][n]['MemFree'] for n in ('0', '1')}
                row['passed'] = all(v >= needed for v in row['free_bytes'].values())
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            row.update(passed=False, error=repr(error))
        if not row['passed']:
            row['reason'] = 'per_node_memory_admission'
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
        if row['stage'] == 'cache_chunk_before_register':
            self.chunk_count += 1
            self.chunk_bytes += row['bytes']
        elif row['stage'] in ('loaded', 'cache_allocated', 'cache_preloaded', 'before_training'):
            self.stages[row['label'] + ':' + row['stage']] = row

    def receipt(self):
        return dict(version=VERSION, adapter_sha256=ADAPTER_SHA, passed=True, per_node_reserve_bytes=RESERVE,
                    admission=self.admissions, initial_admission=self.initial_admission, stages=self.stages,
                    cache_chunks_checked=self.chunk_count, cache_bytes_checked=self.chunk_bytes,
                    reclaim_attempts=self.reclaims)


def valid(receipt):
    try:
        if (receipt['version'] != VERSION or receipt['passed'] is not True
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
                    or set(row['free_bytes']) != {'0', '1'}
                    or any(type(v) is not int or v < needed for v in row['free_bytes'].values())):
                return False
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
        return (type(receipt['cache_chunks_checked']) is int
                and receipt['cache_chunks_checked'] == (sizes['cpu_features'] + CHUNK - 1) // CHUNK
                and receipt['cache_bytes_checked'] == sizes['cpu_features'])
    except (KeyError, TypeError, ValueError):
        return False
