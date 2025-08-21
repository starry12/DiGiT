# L40 igb_small
sudo CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-block-bench-cache \
--node_sequence /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes_IGB_small_0.npy \
--batch_size /mnt/n2/igb_datasets/1024_10_5_5/Batch-Nodes-size_IGB_small_0.npy \
--page_size 4096 --num_page $((1*1000*1000)) --cache_size $((4*1024)) \
--num_queues 128 --queue_depth 1024 --libnvmName /dev/libnvm0


# L40 igb_large
# batch num: 58586
sudo CUDA_VISIBLE_DEVICES=1 ./build/bin/nvm-block-bench-cache \
--node_sequence /mnt/n2/igb_large/Batch-Nodes_IGB_large.npy \
--batch_size /mnt/n2/igb_large/Batch-Nodes-size_IGB_large.npy \
--page_size 4096 --num_page $((1*1000*1000*100)) --cache_size $((4*1024)) \
--num_queues 128 --queue_depth 1024 --libnvmName /dev/libnvm0


# L40 write igb_large
sudo CUDA_VISIBLE_DEVICES=1 ./nvm-readwrite_stripe-bench-jc \
--input /mnt/n2/igb_large/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0 --ioffset 128

# L40 write igb_small
sudo CUDA_VISIBLE_DEVICES=1 ./nvm-readwrite_stripe-bench-jc \
--input /mnt/n2/igb_datasets/small/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0 --ioffset 128