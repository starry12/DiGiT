"""Small raw files exercise the exact sampler-to-bounded-reader boundary."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from candidates.group_cache_migration_v1.common import ROOT

for folder in (ROOT/'ae/igb', ROOT/'ae/igb/runtime'):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
from bounded_io import source, identity
from .raw_adjacency import RawAdjacencySource


def raw(folder, name, values):
    a = np.asarray(values, dtype='<i8')
    p = folder/name
    a.tofile(p)
    spec = dict(path=str(p), offset=0, length=a.nbytes, shape=list(a.shape),
                dtype='<i8', format='raw', identity=list(identity(p.stat())),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return np.memmap(p, mode='r', dtype='<i8'), spec


class RawAdjacencyTests(unittest.TestCase):
    def test_raw_copy_is_bounded_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as td:
            a, spec = raw(Path(td), 'units.i64', range(101))
            reader = RawAdjacencySource(a, spec)
            self.assertIs(source(reader), reader)
            dest = torch.empty(101, dtype=torch.int32)
            digest = reader.copy_to(dest, chunk_bytes=32, limit=2**31)
            np.testing.assert_array_equal(dest.numpy(), a)
            self.assertEqual(digest, spec['sha256'])
            self.assertLessEqual(reader.stats['max_read_bytes'], 32)
            self.assertLessEqual(reader.stats['max_copy_elements'], 4)
            # The shared NPY interface must continue to reject unlabelled raw files.
            with self.assertRaisesRegex(ValueError, 'magic string'):
                source(a)
            np.save(Path(td)/'valid.npy', np.arange(4, dtype=np.int64))
            source(np.load(Path(td)/'valid.npy', mmap_mode='r')).validate(limit=4)

    def test_bad_descriptor_mapping_and_digest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a, spec = raw(Path(td), 'ptr.i64', [0, 1, 2])
            for field, value in [('format', 'npy'), ('dtype', '<i4'), ('offset', 8),
                                 ('length', 25), ('shape', [4]), ('identity', [0]*5)]:
                with self.subTest(field=field):
                    bad = dict(spec, **{field: value})
                    with self.assertRaises(ValueError): RawAdjacencySource(a, bad)
            with self.assertRaisesRegex(ValueError, 'Complete read-only'):
                RawAdjacencySource(a[1:], spec)
            bad = dict(spec, sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                RawAdjacencySource(a, bad).copy_to(torch.empty(3, dtype=torch.int64))

    def test_changed_file_and_integer_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            a, spec = raw(folder, 'units.i64', [0, 2**31])
            with self.assertRaisesRegex(ValueError, 'overflow'):
                RawAdjacencySource(a, spec).copy_to(torch.empty(2, dtype=torch.int32), limit=2**31)
            reader = RawAdjacencySource(a, spec)
            (folder/'replacement').write_bytes(b'\0'*16)
            (folder/'replacement').replace(folder/'units.i64')
            with self.assertRaisesRegex(ValueError, 'changed before read'):
                list(reader.chunks())

    def test_sampler_uses_raw_adapter_and_original_metadata_validation(self):
        from .adjacency_sampler import MigrationSampler
        from uva_sampler import validate_metadata
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            ptr, ps = raw(folder, 'ptr.i64', [0, 2, 3, 3, 3])
            idx, ix = raw(folder, 'units.i64', [3, 1, 0])
            arrays = dict(reordered_indptr=ptr, reordered_indices=idx,
                          group_members=np.array([[0, 2]], dtype=np.int64),
                          group_storage_base=np.array([3]), supernode_to_group=np.array([0]),
                          node_to_primary_row=np.arange(3), storage_to_node=np.array([0, 1, 2, 0, 2]))
            csc = (torch.tensor([0, 2, 3, 3]), torch.tensor([0, 2, 1]), torch.arange(3))
            graph = SimpleNamespace(device=torch.device('cpu'), is_pinned=lambda: True,
                                    ndata={}, edata={}, adj_tensors=lambda _: csc)
            sampler = object.__new__(MigrationSampler)
            sampler.bundle = SimpleNamespace(arrays=arrays, manifest=dict(files={
                'reordered_indptr': ps, 'reordered_indices': ix}))
            sampler._bound_graph = None
            sampler._cuda_metadata = {}
            sampler.num_nodes, sampler.num_groups, sampler.group_size = 3, 1, 2
            sampler.compact, sampler.host_cap, sampler.chunk_bytes = True, 1024, 16

            def metadata(values, n, ng, g, rows, device, compact, cap, **kw):
                sources, total = validate_metadata(values, n, ng, g, rows, compact, cap, 16, shared=True)
                self.assertIs(kw['graph'], graph)
                for name, expected, dtype in [('reorganized_indptr', ptr, torch.int64),
                                              ('reorganized_indices', idx, torch.int32)]:
                    self.assertIsInstance(sources[name], RawAdjacencySource)
                    dest = torch.empty(expected.size, dtype=dtype)
                    sources[name].copy_to(dest, chunk_bytes=16)
                    np.testing.assert_array_equal(dest.numpy(), expected)
                return dict(total=total)

            with patch('candidates.ig_demand_pair_v1.adjacency_sampler.UVAMetadata', side_effect=metadata) as build:
                first = sampler._ensure_cuda_metadata(graph, torch.device('cuda:0'))
                self.assertIs(first, sampler._ensure_cuda_metadata(graph, torch.device('cuda:0')))
                self.assertEqual(build.call_count, 1)
                with self.assertRaisesRegex(ValueError, 'cannot rebind'):
                    sampler._ensure_cuda_metadata(copy.copy(graph), torch.device('cuda:0'))


def gpu_fixture():
    """Startup-only native check: tiny anonymous buffers, no datasets or SSD."""
    import dgl
    from digit.artifacts import ArtifactBundle
    from .adjacency_sampler import MigrationSampler
    from .sampler import configure
    configure('full')
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        ptr, ps = raw(folder, 'ptr.i64', [0, 1, 2, 2, 2])
        idx, ix = raw(folder, 'units.i64', [3, 0])
        arrays = dict(reordered_indptr=ptr, reordered_indices=idx,
                      group_members=np.array([[0, 2]], dtype=np.int64),
                      group_storage_base=np.array([3]), supernode_to_group=np.array([0]),
                      node_to_primary_row=np.arange(3), storage_to_node=np.array([0, 1, 2, 0, 2]))
        manifest = dict(dataset=dict(num_nodes=3), grouping=dict(num_groups=1, group_size=2),
                        files=dict(reordered_indptr=ps, reordered_indices=ix))
        csc = (torch.tensor([0, 2, 3, 3]), torch.tensor([0, 2, 0]), torch.arange(3))
        graph = dgl.graph(('csc', csc), num_nodes=3, idtype=torch.int64).formats('csc')
        graph.pin_memory_()
        sampler = MigrationSampler([2], ArtifactBundle(folder, manifest, arrays), host_cap=1024, chunk_bytes=16)
        try:
            metadata = sampler._ensure_cuda_metadata(graph, torch.device('cuda:0'))
            for name, expected in [('reorganized_indptr', ps), ('reorganized_indices', ix)]:
                actual = metadata.accounting['arrays'][name]
                if actual['source_sha256'] != expected['sha256'] or actual['copy_stats']['max_read_bytes'] > 16:
                    raise RuntimeError('Native raw metadata digest/chunk bound differs')
            output = metadata.sample(torch.arange(3, device='cuda', dtype=torch.int64), 2, 0)
            torch.cuda.synchronize()
            src, rows, flags, eids, groups, counts = [x.cpu().numpy() for x in output]
            np.testing.assert_array_equal(groups, [1, 0, 0])
            np.testing.assert_array_equal(counts, [2, 1, 0])
            np.testing.assert_array_equal(src[:3], [0, 2, 0])
            np.testing.assert_array_equal(rows[:3], [3, 4, 0])
            np.testing.assert_array_equal(eids[:3], [0, 1, 2])
            np.testing.assert_array_equal(flags[:3], [1, 1, 0])
            # Exercise the actual override through DGL block construction and
            # compare against the retained original torch annotation.
            from digit.sampler import DiGiTNeighborSampler, DIGIT_STORAGE_ROW, DIGIT_STORAGE_IS_GROUP
            _, _, blocks = sampler.sample_blocks(graph, torch.tensor([0,1],device='cuda'))
            block=blocks[0]
            optimized_rows=block.srcdata[DIGIT_STORAGE_ROW].clone()
            optimized_flags=block.srcdata[DIGIT_STORAGE_IS_GROUP].clone()
            DiGiTNeighborSampler._attach_cuda_storage_rows(sampler,block,metadata)
            if not torch.equal(optimized_rows,block.srcdata[DIGIT_STORAGE_ROW]) or not torch.equal(optimized_flags,block.srcdata[DIGIT_STORAGE_IS_GROUP]):
                raise RuntimeError('Integrated sampler annotation differs from original')
            sampler.check_window()
        finally:
            torch.cuda.synchronize()
            sampler.close()
            graph.unpin_memory_()
    return 1
