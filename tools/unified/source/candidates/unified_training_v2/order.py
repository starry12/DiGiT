
import hashlib
import numpy as np

def roots_digest(parts):
    h = hashlib.sha256()
    for roots in parts:
        h.update(np.asarray(roots, dtype='<i8').tobytes())
    return h.hexdigest()

def epoch_batches(canonical, epoch, batch_size=1024, order_seed=0, block_nodes=256):
    if epoch < 0 or batch_size < 1 or block_nodes < 1 or batch_size % block_nodes or canonical.ndim != 1 or canonical.dtype != np.int64 or not len(canonical):
        raise ValueError('Expected nonempty verified int64 canonical roots')
    complete = len(canonical)//block_nodes
    indices = np.random.default_rng(np.random.SeedSequence([order_seed, epoch])).permutation(complete)
    per_batch = batch_size//block_nodes
    for start in range(0,complete,per_batch):
        group=indices[start:start+per_batch]
        parts=[canonical[int(index)*block_nodes:(int(index)+1)*block_nodes] for index in group]
        if len(group)<per_batch and len(canonical)%block_nodes:
            parts.append(canonical[complete*block_nodes:])
        yield np.concatenate(parts)
    if complete%per_batch==0 and len(canonical)%block_nodes:
        yield canonical[complete*block_nodes:]

def arm_batches(canonical, config, epoch, arm):
    if arm not in ('gids','digit') or epoch<0 or canonical.ndim!=1 or canonical.dtype!=np.int64 or not len(canonical):
        raise ValueError('Invalid root-order request')
    size=config['batch_size'];order=config['order']
    if arm=='gids':
        roots=np.random.default_rng(np.random.SeedSequence([order['seed'],epoch])).permutation(canonical)
        for start in range(0,len(roots),size):
            yield roots[start:start+size]
    else:
        yield from epoch_batches(canonical,epoch,size,order['seed'],order['block_nodes'])

def order_policy(config,arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return dict(policy='random' if arm=='gids' else 'bfs_chunk_shuffle',
                block_nodes=1 if arm=='gids' else config['order']['block_nodes'],
                seed=config['order']['seed'],base='verified_canonical_roots',tail='retain')

def order_memory_bytes(count,arm,block_nodes=256,batch_size=1024):
    if count<1 or arm not in ('gids','digit'):raise ValueError('Invalid root-order budget')
    return count*8 if arm=='gids' else (count//block_nodes)*8+batch_size*8

def windows(canonical, config, epoch, arm):
    from itertools import islice
    mode = config['execution']; size = config['batch_size']
    batches = arm_batches(canonical, config, epoch, arm)
    if mode['measured_batches']:
        if len(canonical) < size*(mode['warmup_batches']+mode['measured_batches']):
            raise ValueError('Short comparison requires 320 complete root batches')
        warm = list(islice(batches, mode['warmup_batches']))
        yield 'warmup', warm
        remaining = mode['measured_batches']
        while remaining:
            count = min(config['measurement_window_batches'], remaining)
            yield 'training', list(islice(batches, count))
            remaining -= count
    else:
        while True:
            group = list(islice(batches, config['measurement_window_batches']))
            if not group:
                break
            yield 'training', group
