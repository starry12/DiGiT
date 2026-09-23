# Environment

The prepared server provides `/srv/digit-ae/env/bin/python`. Recorded software is Ubuntu 22.04.5, Linux 5.15.0-174-generic, NVIDIA driver 550.54.14, Python 3.8.20, PyTorch 2.2.1 with CUDA runtime 12.1, DGL 2.3.0+cu121, NumPy 1.24.4, CUDA compiler 12.4.99, GCC 11.4.0 and CMake 3.26.4. The native target is NVIDIA L40 / sm89.

To create a separate user-space environment, supply a fresh absolute prefix:

```bash
bash environment/create.sh /absolute/path/to/new-env
export DIGIT_PYTHON=/absolute/path/to/new-env/bin/python
bash run.sh environment
bash run.sh example --output results/cpu_example_01
```

The [Conda lock](../environment/conda-linux-64.lock) and [hashed pip requirements](../environment/requirements.lock) define dependencies; [observed versions](../environment/observed.json) support exact checks. No dataset is downloaded by these commands.

The local CPU checks use the prepared environment; an independent fresh environment installation has not been demonstrated by this packaging step. Native SSD access additionally requires a kernel-matched libnvm module, the intended NVMe controller and GPU/NVMe access. Host setup is administrator-managed. A Python environment alone is insufficient. Follow [NATIVE_BUILD](NATIVE_BUILD.md) for a separate source build; do not replace the active server environment.
