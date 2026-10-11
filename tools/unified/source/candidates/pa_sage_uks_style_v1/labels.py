"""Convert real PA class labels without silently truncating invalid values."""
import numpy as np


def classification_targets(labels, roots, classes):
    import torch
    values = np.asarray(labels).reshape(-1)[roots]
    if values.ndim != 1 or values.dtype.kind not in 'fiu':
        raise ValueError('Expected one numeric PA class label per root')
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError('PA class labels must be finite integers')
    if classes <= 0 or np.any(values < 0) or np.any(values >= classes):
        raise ValueError('PA class label outside configured class range')
    return torch.from_numpy(np.array(values, dtype=np.int64, copy=True))
