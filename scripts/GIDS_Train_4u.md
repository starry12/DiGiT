> 工作路径为 `eveluation` 文件夹下

# 消除cache影响 每次测试前运行3次

```bash
for i in {1..3}; do
  CUDA_VISIBLE_DEVICES=1 ../bam/build/bin/nvm-block-bench-jc \
  --page_size 4096 --pages 7864320 --threads 7864320 \
  --queue_depth 16 --num_queues 16 \
  --random false --access_type 0 \
  --libnvmName /dev/libnvm0
done
```
# Train

#### IGB Small

```bash
CUDA_VISIBLE_DEVICES=1 sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py --dataset_size small --path /mnt/n2/igb_datasets/ \
--epochs 1  --batch_size 1024  --data IGB --uva_graph 1 \
--cache_dim 1024 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((1*1000*1000*1024)) --page_size 4096 \
--pin_file /mnt/n2/igb_datasets/small/processed/paper/pr_small.pt \
--cpu_buffer --cpu_buffer_percent 0.2   --accumulator \
--window_buffer --wb_size 8
```

