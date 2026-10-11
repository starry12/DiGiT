"""Label-independent BFS chunk mixing; NumPy only, no CUDA or graph loading."""
import hashlib
import numpy as np


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def mix_bfs_chunks(order, batch_size, chunk_size, seed, epoch):
    """Rotate the BFS list, shuffle full contiguous chunks, then keep the remainder.

    The RNG is private to (seed, epoch). Rotation changes chunk boundaries and
    which nodes form the short remainder across epochs. Full chunks are assigned
    to batches by a fresh global permutation. Each node appears exactly once.
    No labels, features or validation outcomes are consulted. A chunk crossing
    the circular wrap point need not be contiguous in the original linear list.
    """
    for name, value in [('batch_size',batch_size),('chunk_size',chunk_size),
                        ('seed',seed),('epoch',epoch)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError(name+' must be an integer')
    if batch_size<=1 or chunk_size<=0 or chunk_size>=batch_size or batch_size%chunk_size:
        raise ValueError('chunk_size must be a positive proper divisor of batch_size')
    if seed<0 or epoch<0:
        raise ValueError('seed and epoch must be nonnegative')
    values=np.asarray(order)
    if values.ndim!=1 or values.dtype!=np.int64:
        raise ValueError('order must be a one-dimensional int64 array')
    # The caller validates uniqueness/coverage against the training split.
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(epoch),0xB15]))
    offset=int(rng.integers(len(values))) if len(values) else 0
    rotated=np.roll(values,-offset)
    full_chunks=len(values)//chunk_size
    end=full_chunks*chunk_size
    permutation=rng.permutation(full_chunks)
    mixed=np.empty_like(values)
    mixed[:end]=rotated[:end].reshape(full_chunks,chunk_size)[permutation].reshape(-1)
    mixed[end:]=rotated[end:]
    return mixed,dict(policy='bfs_mixed_chunks',seed=int(seed),epoch=int(epoch),
        batch_size=int(batch_size),chunk_size=int(chunk_size),
        chunks_per_full_batch=int(batch_size//chunk_size),rotation_offset=offset,
        complete_chunks=int(full_chunks),remainder_nodes=int(len(values)-end),
        remainder_policy='kept last after epoch-specific circular rotation',
        chunk_permutation_sha256=digest(permutation.astype(np.int64)),
        ordered_nodes_sha256=digest(mixed))
