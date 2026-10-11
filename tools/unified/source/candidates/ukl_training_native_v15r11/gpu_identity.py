"""Bind physical selection by full UUID and verify the actual current context."""
import ctypes as C
import os
import uuid


def selection_from_env(expected=None):
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    index = os.environ.get('UKL_SELECTED_GPU_INDEX', '')
    if index not in ('0', '1', '2', '3') or not value.startswith('GPU-'):
        raise RuntimeError('Full GPU UUID and physical index required before Python starts')
    try:
        canonical = 'GPU-'+str(uuid.UUID(value[4:]))
    except ValueError:
        raise RuntimeError('Invalid full GPU UUID') from None
    if canonical != value:
        raise RuntimeError('Canonical full GPU UUID required')
    selected = (int(index), value)
    if expected is not None and selected != tuple(expected):
        raise RuntimeError('GPU selection differs from admitted controller')
    return selected


def verify_cuda_uuid(expected):
    # cuDeviceGetUuid is an exported Driver API; cudaDeviceGetUuid was not an
    # exported function in the host libcudart. Call only after model CUDA init.
    driver = C.CDLL('libcuda.so.1')
    for name, args in [('cuDeviceGetCount', [C.POINTER(C.c_int)]),
                       ('cuCtxGetCurrent', [C.POINTER(C.c_void_p)]),
                       ('cuCtxGetDevice', [C.POINTER(C.c_int)]),
                       ('cuDeviceGetUuid', [C.c_void_p, C.c_int])]:
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = args, C.c_int
    def checked(name, *args):
        result = getattr(driver, name)(*args)
        if result != 0:
            raise RuntimeError(name+' failed: '+str(result))
    count, device, context = C.c_int(), C.c_int(-1), C.c_void_p()
    checked('cuDeviceGetCount', C.byref(count))
    checked('cuCtxGetCurrent', C.byref(context))
    if count.value != 1 or not context.value:
        raise RuntimeError('Exactly one visible CUDA device and a current context required')
    checked('cuCtxGetDevice', C.byref(device))
    if device.value != 0:
        raise RuntimeError('Current CUDA context must be logical device zero')
    raw = (C.c_ubyte*16)()
    checked('cuDeviceGetUuid', C.byref(raw), device.value)
    actual = 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))
    if actual != expected:
        raise RuntimeError('CUDA context UUID differs from admitted GPU')
    return dict(passed=True, uuid=actual, logical_device=0, visible_devices=count.value)
