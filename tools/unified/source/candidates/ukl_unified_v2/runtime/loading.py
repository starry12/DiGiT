def load_checked(arena, callback, arm, placement):
    result = arena.load(callback)
    if arm == 'digit':
        placement(arena, 'loaded')
    return result
