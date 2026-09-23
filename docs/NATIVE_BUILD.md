# Building this source snapshot

The prepared server is the currently validated native execution route. The concise source layout and recipe below need their own fresh native acceptance; do not substitute older server results for that evidence.

Use the locked Python environment, CUDA toolkit and sm89-compatible hardware described in [ENVIRONMENT](ENVIRONMENT.md). Inspect the build plan first:

```bash
export DIGIT_PYTHON="$HOME/digit-env/bin/python"
export DIGIT_CMAKE="$HOME/digit-env/bin/cmake"
export DIGIT_NVCC=/usr/local/cuda-12.4/bin/nvcc
bash scripts/build_native.sh --output "$HOME/digit-native" --dry-run
bash scripts/build_native.sh --output "$HOME/digit-native"
```

Adjust the environment and CUDA paths to your installation. The output directory must be new and outside the source checkout. The script copies into a new directory and builds libnvm user space, the SSD identity utility, the feature store with useful I/O counters, the bidirectional DiGiT sampler, and output annotation. It does not install a kernel module or access the SSD. The source directory is preserved. Build logs are under the new copy's `results/build/`. The new copy receives its own manifest.

Switch to the built copy before binding data or launching native commands:

```bash
cd "$HOME/digit-native"
bash run.sh verify
```

After administrator-managed host preparation and [read-only data binding](DATA.md), use a fresh output path for each step. These administrative native commands require the correct privileges and tmux; the reviewer account instead uses the fixed `digit-ae` service.

```bash
bash run.sh check --model sage --gpu 2 --output results/check_sage_01
bash run.sh smoke --model sage --gpu 2 --output results/smoke_sage_01
bash run.sh representative --model sage --gpu 2 --output results/full_sage_01
bash run.sh summarize --model sage --input results/full_sage_01 --output results/summary_sage_01
```

Substitute `gcn` or `gat`. The representative action repeats both smoke arms before fresh full training. All native models share exclusive locks. A build, a filesystem preflight, a paired smoke and full training are distinct stages. Do not update the active service or its dataset bindings while another experiment runs.
