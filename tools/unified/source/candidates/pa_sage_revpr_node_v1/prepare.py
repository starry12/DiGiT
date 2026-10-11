"""Invoked only after the controller owns the AE/SSD exclusion locks."""
from .common import *


def prepare(folder, execution):
    import numpy as np
    from candidates.pa_sage_bidir_native_v2.run import input_binding, identity
    from candidates.pa_sage_bidir_native_v2.common import verify as parent_verify
    from candidates.pa_sage_cache_v1.selection import top, page_rows
    # Rehash the AE-bound inputs at idle time; do not weaken historical identity checks.
    binding = input_binding(folder, parent_verify())
    p = cfg()
    scores_path = ROOT / p['revpr_scores']
    before = identity(scores_path)
    require(sha(scores_path) == p['revpr_scores_sha256'], 'RevPR score provenance differs')
    scores = np.load(scores_path, mmap_mode='r')
    require(scores.shape == (p['graph']['nodes'],) and scores.dtype == np.float64,
            'Wrong RevPR scores')
    # Verify the old page choices from the SAME scores before changing selection granularity.
    pages, _ = page_rows(scores, p['cpu_cache_rows'])
    old = np.load(ROOT / p['base_layout'] / 'gids_cpu_rows.npy', mmap_mode='r')
    require(np.array_equal(pages, old), 'Scores do not reproduce current AE page-adapted cache')
    del pages
    hot, threshold = top(scores, p['cpu_cache_rows'])
    require(identity(scores_path) == before, 'Scores changed during selection')
    dest = folder / 'prepared'
    dest.mkdir()
    hot_path = dest / 'node_hot.npy'
    with hot_path.open('xb') as f:
        np.save(f, hot, allow_pickle=False)
    from ae.common import check_payload
    storage_rows = check_payload('papers_gids')['bytes'] // 512
    binding['parent_candidate_sha256'] = binding['candidate_sha256']
    binding.update(candidate_sha256=execution, protocol_sha256=sha(P))
    binding['node_cache'] = dict(path=str(hot_path), sha256=sha(hot_path),
        rows=len(hot), feature_bytes=len(hot) * 512, storage_rows=storage_rows,
        score_path=str(scores_path), score_sha256=p['revpr_scores_sha256'],
        threshold=float(threshold), selection='descending RevPR score; ascending node ID ties',
        old_page_selection_reproduced=True)
    for path in (scores_path, hot_path):
        binding['files'][str(path)] = dict(sha256=sha(path), identity=identity(path))
    write(folder / 'inputs.json', binding)
    return binding
