"""Bind reusable directed CSC, official split, original feature data and ranking."""
from .common import *


def main():
    import numpy as np
    import torch
    from ae.pa_sage.common import source_config, splits, csc_paths
    from ae.common import check_payload
    require(not (OUT / 'inputs.json').exists(), 'Preserve previous input binding')
    p = read(HERE / 'protocol.json')
    cfg = source_config()
    files = {}

    def bind(path, expected=None, inherited=None):
        path = Path(path)
        before = identity(path)
        if inherited:
            require(before == inherited['identity'], 'Inherited source identity changed')
            digest = inherited['sha256']
            method = 'accepted checksum plus unchanged device/inode/size/mtime/ctime'
        else:
            progress(OUT, 'hash_input', path=str(path))
            digest = sha(path)
            method = 'full file SHA256 this preparation'
        require(expected is None or expected == digest, 'Source checksum mismatch: ' + str(path))
        require(before == identity(path), 'File changed while hashing')
        files[str(path)] = dict(identity=before, sha256=digest, verification=method)

    old_path = ROOT / 'results/pa_sage_ablation_cache_20260924_v3/resume2/native/inputs.json'
    old = read(old_path)
    for key in ('source_features', 'label_identity'):
        d = cfg[key]
        bind(d['path'], d['sha256'], old['files'][d['path']])
    ptr, idx = csc_paths()
    for path in (ptr, idx):
        bind(path, cfg['source_contract']['csc'][path.name]['sha256'])
    ip, ix = [np.load(x, mmap_mode='r') for x in (ptr, idx)]
    require(ip.shape == (p['nodes'] + 1,) and ix.shape == (p['edges'],) and
            ip.dtype == ix.dtype == np.int64 and int(ip[0]) == 0 and int(ip[-1]) == len(ix),
            'Wrong directed CSC')
    bind(cfg['splits']['train']['path'], cfg['splits']['train']['sha256'])
    train = splits('train')
    require(len(train) == p['train_count'], 'Wrong official train count')
    bind(p['pin_file'])
    rank = torch.load(p['pin_file'], map_location='cpu')
    require(rank.ndim == 1 and rank.dtype == torch.int64 and len(rank) >= p['cpu_cache_rows'],
            'Ranking must contain logical int64 node IDs')
    hot = rank[:p['cpu_cache_rows']].numpy()
    require(hot.min() >= 0 and hot.max() < p['nodes'] and len(np.unique(hot)) == len(hot),
            'Invalid RevPR Top-K')
    payload = check_payload('papers_gids')
    require(payload['offset'] % 512 == 0 and payload['bytes'] >= p['nodes'] * 512,
            'Original SSD payload cannot cover requested logical rows')
    bind(payload['state'], payload['state_sha256'])
    binding = dict(files=files, csc=dict(indptr=str(ptr), indices=str(idx)),
        features=cfg['source_features']['path'], labels=cfg['label_identity']['path'],
        train_split=cfg['splits']['train']['path'],
        train_sorted_sha256=hashlib.sha256(np.sort(train).tobytes()).hexdigest(),
        hot_rows=p['cpu_cache_rows'], hot_rows_sha256=hashlib.sha256(hot.tobytes()).hexdigest(),
        rank_length=len(rank), inherited_feature_binding=dict(path=str(old_path), sha256=sha(old_path)),
        ssd_payload=payload, graph_assumption=p['graph_limit'], prepared_unix=time.time())
    check_inputs(binding)
    write(OUT / 'inputs.json', binding)
    progress(OUT, 'inputs_prepared', train_count=len(train), cpu_cache_rows=len(hot))


if __name__ == '__main__':
    main()
