# Environment setup from scratch

Use this guide to create a separate environment from the source checkout. The path is: install host prerequisites, get the code, create the locked Python environment, run the CPU checks, build native components, then bind prepared data and run native acceptance. If you use the provided AE server, its environment is already installed; follow the [reviewer guide](REVIEWER.md).

## 1. Prepare the host

The dependency locks target **Linux x86_64**. Install Git, Bash, Python 3 and Conda first. If Conda is unavailable, follow its [official installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/). `conda --version` must work in the shell used below; alternatively set `DIGIT_CONDA` to the absolute path of the Conda executable.

The recorded server stack is listed below. These are the reference versions used by the existing environment, rather than a claim that all other versions have been tested.

| Component | Recorded version |
|---|---|
| OS / kernel | Ubuntu 22.04.5 / Linux 5.15.0-174-generic |
| GPU / native compilation target | NVIDIA L40 / sm89 |
| NVIDIA driver | 550.54.14 |
| CUDA Toolkit compiler | 12.4.99 |
| GCC / CMake | 11.4.0 / 3.26.4 |
| Python | 3.8.20 |
| PyTorch / its CUDA runtime | 2.2.1 / 12.1 |
| DGL | 2.3.0+cu121 |
| NumPy / pybind11 | 1.24.4 / 2.13.6 |

For native compilation, install the CUDA Toolkit and a compatible NVIDIA driver using the [CUDA 12.4 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-linux/index.html), plus a C/C++ compiler and Make. The Conda lock supplies CMake; the build guide selects its executable explicitly. The CUDA runtime supplied with PyTorch and the host `nvcc` compiler are recorded separately: installing the Python environment does not install the host CUDA Toolkit or GPU driver.

Check the host tools before a native build:

```bash
conda --version
nvidia-smi
/usr/local/cuda-12.4/bin/nvcc --version
gcc --version
make --version
```

Adjust the CUDA path to the actual installation. The current native build targets sm89; other GPU architectures need a separate build and validation.

## 2. Get the source

```bash
git clone https://github.com/starry12/DiGiT.git
cd DiGiT
bash run.sh verify
bash run.sh matrix
```

`verify` checks the source manifest and should report `verified: true`. `matrix` lists the available experiment configurations. For the frozen initial AE snapshot, check out `ae-pa-v1` after cloning; later documentation updates live on `main`.

Run the following commands from the repository root unless a step explicitly changes directory.

## 3. Create the locked Python environment

Choose a new absolute directory outside the checkout. The installer refuses an existing prefix.

```bash
bash environment/create.sh "$HOME/digit-env"
export DIGIT_PYTHON="$HOME/digit-env/bin/python"
```

The installer creates Python and the base packages from the [explicit Conda lock](../environment/conda-linux-64.lock), installs the [hashed pip requirements](../environment/requirements.lock), runs `pip check`, and checks package versions and imports against [the recorded environment](../environment/observed.json). It requires network access to the package locations in the locks. The selected DGL CUDA wheel is included in the pip requirements; this recipe does not require a separate DGL source build.

`DIGIT_PYTHON` selects the interpreter used by `run.sh`; it avoids dependence on which Conda environment is active. Set it again in a new shell. The installer does not download graph data, compile the project's native extensions, or configure SSD access.

## 4. Check the environment and run the CPU example

```bash
bash run.sh environment
bash run.sh example --output results/cpu_example_01
```

The environment check should report `passed: true` and an empty `issues` list. The CPU example should report `passed: true` for GraphSAGE, GCN and GAT, with three optimizer updates each. It needs no GPU, dataset or raw SSD. Choose a fresh output directory when repeating it.

These checks cover package imports and a small model/optimizer execution. The environment report records `native_device_io_tested: false`; CPU success does not establish native SSD readiness.

## 5. Compile the native components

Keep the selected Python interpreter, select the locked CMake executable, and point to the host CUDA compiler:

```bash
export DIGIT_CMAKE="$HOME/digit-env/bin/cmake"
export DIGIT_NVCC=/usr/local/cuda-12.4/bin/nvcc
bash scripts/build_native.sh --output "$HOME/digit-native" --dry-run
bash scripts/build_native.sh --output "$HOME/digit-native"
```

`$HOME/digit-native` must be a new directory outside the source checkout. The script builds into that copy and leaves the source checkout intact. It builds libnvm user space, the SSD identity utility, the instrumented feature store, the DiGiT sampler and output annotation. It neither installs a kernel module nor accesses the SSD. Build logs are written to the copy's `results/build/build.log`.

See the [native build guide](NATIVE_BUILD.md) for details and the subsequent execution commands. After a successful build, run data binding and native commands from the **built copy**.

## 6. Prepare host storage access, bind data and validate

Native training additionally needs the correct GPU/NVMe access, a libnvm module built for the running kernel and NVIDIA driver, and prepared graph/feature data with matching SSD verification receipts. The administrator should review the vendored [BaM hardware and host setup instructions](../third_party/bam/README.md) against the actual host. Its generic examples do not define this artifact's selected dataset or device bindings.

Follow the [data guide](DATA.md) to register the prepared read-only inputs in the built copy, then the [native build guide](NATIVE_BUILD.md) to run `check`, paired `smoke`, and the full `representative` comparison. The dataset download alone does not create the selected layout, traces or SSD contents. The prepared AE server already supplies those inputs.

## Validation status

The existing prepared environment has passed the package/import checks and CPU example. A fresh online installation, a native rebuild of this reorganized source, and a complete fresh download-to-layout-to-SSD pipeline have not yet been validated end to end. The currently accepted native experiments use the preserved server release. The instructions above distinguish the available setup recipe from that existing execution evidence.
