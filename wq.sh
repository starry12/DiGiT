# 编译GIDS
su embed
conda activate gids
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


# 编译bam
sudo su
cd /home/embed/Documents/bam-gids/build
rm -rf build
mkdir -p build
sudo chown -R embed:embed /home/embed/Documents/bam-gids
cd build
cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -D NVIDIA=/usr/src/nvidia-550.54.14/ -D driver_include:PATH=/usr/src/nvidia-550.54.14/nvidia -D CMAKE_BUILD_TYPE=RELEASE ..
make libnvm -j$(nproc)
make benchmarks -j$(nproc) 2>/home/embed/Documents/gids/bam/build_err.txt
cd module
make -j$(nproc)


#编译dgl
cd /home/embed/Documents/dgl/
bash script/build_dgl.sh -g
cd python/
python setup.py install
python setup.py build_ext --inplace

# sample test
sudo /home/embed/anaconda3/envs/dgl-dev-gpu-121/bin/python homogenous_train.py \
--dataset_size papers100M --data OGB --num_classes 172 \
--path /mnt/n3/papers100M-bin/processed \
--epochs 1 --batch_size 1024 --uva_graph 1 \
--cache_dim 128 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 128 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((111059956*128)) --page_size 512 

# write igb-small-reorder
./nvm-readwrite_stripe-bench-jc \
--input /home/embed/Downloads/igb_datasets/small/processed/paper/node_feat_reordered.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0

# write igb-small
./nvm-readwrite_stripe-bench-jc \
--input /mnt/n3/igb_datasets/small/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0

# write igb-large
./nvm-readwrite_stripe-bench-jc \
--input /mnt/n3/igb_large/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm2

# write igb-full
./nvm-readwrite_stripe-bench-jc \
--input /mnt/n3/IGBH/full/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0

# write papers100M
./nvm-readwrite_stripe-bench-jc \
--input /mnt/n3/papers100M-bin/processed/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0



#L40 igb-small
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py --dataset_size small --path /mnt/n3/igb_datasets/ \
--epochs 1  --batch_size 1024  --data IGB --uva_graph 1 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((550*1000*1000*1024)) --page_size 4096 \
--pin_file /home/embed/Documents/gids/evaluation/wq/pr_small.pt \
--cpu_buffer --cpu_buffer_percent 0.2   --accumulator \
--window_buffer --wb_size 8

#L40 igb-large
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py --dataset_size large --path /mnt/n3/ \
--epochs 1 --batch_size 1024 --data IGB --uva_graph 1 \
--cache_dim 1024 --log_every 1000 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((550*1000*1000*1024)) --page_size 4096 \
--pin_file /home/embed/Documents/gids/evaluation/pr_large_original_all.pt \
--cpu_buffer --cpu_buffer_percent 0.2 --GIDS --accumulator --window_buffer --wb_size 2

#L40 igb-full
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py --dataset_size full --path /mnt/n3/ \
--epochs 2 --batch_size 1024 --data IGB --uva_graph 1 \
--cache_dim 1024 --log_every 1000 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((550*1000*1000*1024)) --page_size 4096 \
--pin_file /home/embed/Documents/gids/pr_full.pt \
--cpu_buffer --cpu_buffer_percent 0.1 --GIDS --accumulator --window_buffer --wb_size 2

#L40 papers100M
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py \
--dataset_size papers100M --data OGB --num_classes 172 \
--path /mnt/n3/papers100M-bin/processed \
--epochs 10 --batch_size 1024 --uva_graph 1 \
--cache_dim 128 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 128 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((111059956*128)) --page_size 512 \
--pin_file /mnt/n3/papers100M-bin/processed/pr_papers100M.pt \
--cpu_buffer --cpu_buffer_percent 0.1 --accumulator \
--window_buffer --wb_size 2


#/mnt/n3/papers100M-bin/processed/pr_papers100M.pt
# /home/embed/Documents/gids/evaluation/freq_hotness.pt
# dgl-dev-gpu-121

#L40  UKS
sudo /home/embed/anaconda3/envs/gids/bin/python homogenous_train.py \
--data UKS --path /home/embed/Documents/Hyperion_notes/dataset/ukunion/ \
--epochs 1 --batch_size 1024 --uva_graph 1 \
--cache_dim 256 \
--model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 256 \
--cache_size $((4*1024)) --num_ssd 1 --num_ele $((111059956*128)) --page_size 1024

#--pin_file /mnt/n3/papers100M-bin/processed/pr_papers100M.pt \
#--cpu_buffer --cpu_buffer_percent 0.1 --accumulator \
#--window_buffer --wb_size 2



# baseline
sudo /home/embed/anaconda3/envs/dgl-dev-gpu-121/bin/python homogenous_train_baseline.py --dataset_size small --path /home/embed/Downloads/igb_datasets/ \
--epochs 1  --batch_size 1024  --data IGB --model_type sage --num_layers 3 --fan_out '10,5,5' --emb_size 1024

# pagerank
python page_rank_node_list_gen.py --dataset_size small --path /mnt/n3/igb_datasets/ --out_path ./pr_small.pt --data IGB 
python page_rank_node_list_gen.py --dataset_size large --path /mnt/n3/ --out_path ./pr_large_all.pt --data IGB
python page_rank_node_list_gen.py --path /mnt/n3/papers100M-bin/processed --out_path ./pr_papers100M.pt --data OGB

sudo /home/embed/anaconda3/envs/gids/bin/python page_rank_node_list_gen.py --path /home/embed/ukunion/ukunion --out_path ./pr_uk2014.pt --data UKS


sudo /home/embed/anaconda3/envs/gids/bin/python test_cache.py   --dataset igb260m   --mode offline --device cuda --batch_size 1024 --fanout 10 5 5


sudo scp -P 13029 -r embed@10.102.0.229:/mnt/n2/IGBH/full/processed/paper__cites__paper/edge_index_csc_col_idx.npy /mnt/n3/IGBH/full/processed/paper__cites__paper/


sudo scp -P 60022 -r embed@101.76.222.12:/home/embed/Documents/Hyperion_notes/dataset/clueweb/edge_index.tar.gz /mnt/nvme2n1/clueweb/


ssh -p 2222 embed@101.76.221.100

ssh -p 22 embed@10.102.35.40