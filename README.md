# DiGiT

DiGiT is a high-performance out-of-core GNN system built on top of GIDS. It optimizes GPU-side sampling, SSD-aware feature access, and multi-level caching for billion-scale graphs on a single machine.


## Overview

- **Foundation:** Extends the GIDS codebase and workflow; follows the same build & runtime assumptions.  
- **Goal:** Efficient training when node features do not fit in GPU/CPU memory by streaming from SSD.  
- **Recommended:** CUDA-capable GPU, modern NVIDIA driver, and NVMe SSD(s). GPUDirect Storage (GDS) can further reduce I/O overhead (optional depending on your stack).

---

## Prerequisites

- **OS:** Linux (x86_64)
- **GPU/Driver:** NVIDIA driver compatible with your CUDA toolkit (example below uses CUDA 12.4)
- **Toolchain:** CUDA Toolkit, CMake, GCC/Clang, Python 3.8–3.10, pip/conda
- **Python packages:** DGL (built from source with GPU), PyTorch (CUDA build), other GIDS requirements
- **Storage:** NVMe SSD accessible at `/dev/libnvm0` (or adjust device path)

> Quick checks:
> ```bash
> nvcc --version        # Verify CUDA toolkit
> nvidia-smi            # Verify GPU & driver
> python -V             # Verify Python
> ```

---

## Setup

### 1) Build BAM
Compiles NVMe access libraries and benchmarks used to stripe/write node features to SSD.

```bash
# Build bam
sudo su
cd /home/embed/Documents/bam-gids/build
rm -rf build
mkdir -p build
sudo chown -R embed:embed /home/embed/Documents/bam-gids
cd build
cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
      -D NVIDIA=/usr/src/nvidia-550.54.14/ \
      -D driver_include:PATH=/usr/src/nvidia-550.54.14/nvidia \
      -D CMAKE_BUILD_TYPE=RELEASE ..
make libnvm -j$(nproc)
make benchmarks -j$(nproc) 2>/home/embed/Documents/gids/bam/build_err.txt
cd module
make -j$(nproc)
```

**Notes**
- Adjust CUDA paths, driver include path, and versions to match your system.
- Root privileges are needed only where required (e.g., device access).

---

### 2) Build DGL (GPU)
Build DGL from source with GPU support to match your CUDA/PyTorch.

```bash
cd /home/embed/Documents/dgl/
bash script/build_dgl.sh -g
cd python/
python setup.py install
python setup.py build_ext --inplace
```

**Notes**
- Ensure your PyTorch CUDA version matches your CUDA toolkit.
- If you use a conda env, activate it before building/installing.

---

### 3) Build GIDS Components
Compile and install GIDS modules (DiGiT follows this layout).

```bash
cd /home/embed/Documents/gids/gids_module/build
cmake .. \
  -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
  -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make -j$(nproc)

cd BAM_Feature_Store
pip install .

cd ../../../GIDS_Setup
pip install .

cd ../evaluation
```

**Notes**
- The `pip install` steps register local Python packages used by the training scripts.
- If you see missing libraries at runtime, verify `LD_LIBRARY_PATH` includes CUDA and any custom lib paths.

---

## Run (IGB-small example)

Below is a minimal end-to-end flow with the **IGB-small** dataset. Replace paths, device names, and hyperparameters to fit your environment.

### 1) Preprocess data (graph reorder)
Produces a page- and access-friendly layout for downstream I/O.

```bash
python graph_reorder.py --dataset_size small --path /mnt/n3/igb_datasets/ --data IGB
```

**Notes**
- Ensure `/mnt/n3/igb_datasets/` exists and has sufficient space.
- For other dataset sizes (e.g., `medium`, `large`), change `--dataset_size` accordingly.

---

### 2) Write features to SSD
Stripe node features to the NVMe device with a page-aligned configuration.

```bash
./nvm-readwrite_stripe-bench-jc \
  --input /mnt/n3/igb_datasets/small/processed/paper/node_feat.npy \
  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
  --page_size $((4*4096)) \
  --libnvmName /dev/libnvm0
```

**Notes**
- `--page_size $((4*4096))` sets a 16 KB logical page; tune to your SSD/page size.
- Device path `/dev/libnvm0` may differ on your system.
- High queue/thread values assume sufficient CPU cores and NVMe capability.

---

### 3) Launch training
Runs a 3-layer GraphSAGE model with a modest fan-out on IGB-small.

```bash
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py \
  --dataset_size small --path /mnt/n3/igb_datasets/ \
  --epochs 1 --batch_size 1024 --data IGB --uva_graph 1 \
  --model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 \
  --cache_size $((4*1024)) --num_ssd 1 --num_ele $((550*1000*1000*1024)) --page_size 4096 \
  --pin_file /home/embed/Documents/gids/evaluation/wq/pr_small.pt \
  --cpu_buffer --cpu_buffer_percent 0.2 \
  --accumulator --window_buffer --wb_size 8
```
