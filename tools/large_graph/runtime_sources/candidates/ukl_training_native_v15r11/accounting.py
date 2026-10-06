"""Classify current counters, never subtract a historical cache baseline."""
def classify(memory):
    keys=('file','shmem','anon','file_dirty','file_writeback')
    if any(type(memory.get(k)) is not int or memory[k]<0 for k in keys):
        raise ValueError('Missing or invalid memory counter')
    if memory['shmem']>memory['file']:
        raise ValueError('Inconsistent file/shmem counters')
    return dict(non_shmem_file=memory['file']-memory['shmem'],
                shared=memory['shmem'],anonymous=memory['anon'],
                dirty=memory['file_dirty'],writeback=memory['file_writeback'])
