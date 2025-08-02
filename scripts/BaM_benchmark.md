# 消除cache影响 每次测试前运行3次

```bash
for i in {1..3}; do
  CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-block-bench-jc \
  --page_size 4096 --pages 7864320 --threads 7864320 \
  --queue_depth 16 --num_queues 16 \
  --random false --access_type 0 \
  --libnvmName /dev/libnvm0
done
```

# 数据写入

#### IGB Small 16KB原图写入 

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-readwrite_stripe-bench-jc \
--input /mnt/n2/igb_datasets/small/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0
```

#### IGB Full 16KB原图写入

# 真实Batch读写

#### IGB Small 4KB

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-block-bench-cache-jc \
--node_sequence /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes_IGB_small_0.npy \
--batch_size /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes-size_IGB_small_0.npy \
--page_size 4096 --num_page $((1*1000*1000)) \
--blk_size 128 \
--cache_size $((1024)) \
--num_queues 16 --queue_depth 16 \
--libnvmName /dev/libnvm0
```

#### IGB Small 16KB

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-block-bench-cache-jc \
--node_sequence /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes_IGB_small_0.npy \
--batch_size /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes-size_IGB_small_0.npy \
--page_size $((4096*4)) --num_page $((1*1000*1000/4)) \
--blk_size 128 \
--cache_size $((1024)) \
--num_queues 16 --queue_depth 16 \
--libnvmName /dev/libnvm0
```

# Block Cache

