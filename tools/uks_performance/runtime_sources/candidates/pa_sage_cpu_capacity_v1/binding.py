"""Hash immutable graph/address files once after the grid; workers recheck identities."""
from pathlib import Path
from .common import ROOT, read, sha, require, write, identity, object_sha, progress, heavy_gate


def bind(p, protocol_path, output, execution):
    heavy_gate()
    from candidates.pa_sage_layout_shared_resume_v3.pool import spec, check_receipt
    from candidates.pa_sage_layout_shared_resume_v3.graph_binding import descriptors
    from ae.pa_sage.common import source_config
    base, overlay, data = Path(p['layout']['base']), Path(p['layout']['overlay']), ROOT / p['data']
    m = read(base / 'final/bundle/manifest.json')
    require(sha(base / 'final/bundle/manifest.json') == p['layout']['manifest_sha256'], 'Wrong fixed layout')
    over = read(overlay / 'overlay_receipt.json')
    require(sha(overlay / 'overlay_receipt.json') == p['layout']['overlay_receipt_sha256'] and over['passed'],
            'Invalid reverse-edge overlay')
    build, validation = read(base / 'build_receipt.json'), read(base / 'final/validation.json')
    require(build['passed'] and not build['fixture'] and validation['passed'] and
            validation['expanded_adjacency_multiset_exact'] and validation['storage_inverse_exact'] and
            validation['group_padding_checked'], 'Fixed g2/r20 validation missing')
    require(build['manifest_sha256'] == over['layout_manifest_sha256'] == p['layout']['manifest_sha256'], 'Stale layout receipt')
    require(over['nodes'] == p['graph']['nodes'] and over['edges'] == p['graph']['edges'], 'Overlay graph differs')
    pool, receipt = spec('real'), check_receipt('real')
    require(sha(receipt['receipt_path']) == p['features']['receipt_sha256'] and
            receipt['offset'] == p['features']['pool_offset'] and
            m['feature']['num_storage_rows'] * 512 <= receipt['verified_bytes'], 'Real payload binding differs')
    require(m['files']['reordered_features']['sha256'] == pool['source_full_sha256'], 'Layout does not match resident features')
    files = {}
    def add(path, expected=None, full=True):
        path = Path(path).resolve(); before = identity(path)
        progress(output, 'binding', path=str(path), completed_files=len(files))
        digest = sha(path) if full else None
        require(identity(path) == before and (expected is None or digest == expected), 'Input changed: ' + str(path))
        files[str(path)] = dict(identity=before, sha256=digest, full_hash_checked=full)
    add(protocol_path)
    for f in ('build_receipt.json', 'contract.json', 'final/descriptor.json', 'final/validation.json', 'final/bundle/manifest.json'):
        add(base / f)
    for name, desc in m['files'].items():
        if name != 'reordered_features': add(base / 'final/bundle' / desc['path'], desc['sha256'])
    add(overlay / 'overlay_receipt.json')
    for name, digest in over['files'].items(): add(overlay / name, digest)
    add(data / 'prepared.json')
    csc = descriptors(data, p['graph'])
    for desc in csc.values(): add(desc['path'], desc['file_sha256'])
    add(receipt['receipt_path'], p['features']['receipt_sha256'])
    add(pool['state'], pool['state_sha256'])
    add(pool['source'], full=False) # already covered by complete SSD readback + source identity
    source = source_config()
    for desc in (source['label_identity'], source['splits']['train'], source['source_contract']['original_edges']): add(desc['path'], desc['sha256'])
    add(p['orders_file'])
    # Unreordered logical features are used only by the short value oracle.
    add(source['source_features']['path'], source['source_features']['sha256'])
    result = dict(schema='digit-cache-input-binding-v3', source_sha256=execution,
        protocol_sha256=sha(protocol_path), files=files, graph_sha256=object_sha(p['graph']),
        layout_sha256=p['layout']['manifest_sha256'], feature_receipt_sha256=p['features']['receipt_sha256'],
        source_features=source['source_features']['path'], labels=source['label_identity']['path'],
        original_edges=source['source_contract']['original_edges']['path'],
        train_split=source['splits']['train'], manifest=m, fixture=False, passed=True)
    write(output / 'inputs.json', result)
    return result


def check(binding, protocol_path, execution):
    require(binding['passed'] and binding.get('fixture') is False and binding['source_sha256'] == execution and
            binding['protocol_sha256'] == sha(protocol_path), 'Input binding does not match this execution')
    for path, record in binding['files'].items():
        require(identity(path) == record['identity'], 'Bound input was replaced/modified: ' + path)


def load_bundle(p, binding):
    import numpy as np
    from digit.artifacts import ArtifactBundle
    from candidates.pa_sage_bidir_native_v2.overlay import apply
    root = Path(p['layout']['base']) / 'final/bundle'
    m = binding['manifest']; arrays = {}
    for name, desc in m['files'].items():
        if name == 'reordered_features': continue
        a = np.load(root / desc['path'], mmap_mode='r', allow_pickle=False)
        require(a.dtype == np.int64 and list(a.shape) == desc['shape'], 'Wrong metadata header: ' + name)
        arrays[name] = a
    original = ArtifactBundle(root, m, arrays)
    overlay = Path(p['layout']['overlay'])
    return apply(original, np.load(overlay / 'reordered_indptr.npy', mmap_mode='r'),
                 np.load(overlay / 'reordered_indices.npy', mmap_mode='r'), p['graph']['edges'])
