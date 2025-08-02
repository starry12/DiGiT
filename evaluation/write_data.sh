
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/nvme17/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1

../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/author/node_feat.npy  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((269346174*4096))

../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/fos/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((546567057*4096)) 


../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/institute/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((547280017 * 4096)) 

lsblk

# compile bam
sudo su
cd /home/embed/Documents/GIDS/bam/build
cmake ..
make libnvm -j$(nproc)
make benchmarks -j$(nproc)


cd /home/embed/Documents/GIDS/bam/build
make readwrite_stripe-benchmark

dmesg | grep nvme3
echo -n "0000:b3:00.0" > /sys/bus/pci/devices/0000\:b3\:00.0/driver/unbind # 1T
echo -n "0000:1b:00.0" > /sys/bus/pci/devices/0000\:1b\:00.0/driver/unbind # 7T
cd /home/embed/Documents/GIDS/bam/build/module
sudo make load
## list /dev named libnvm*
ls /dev/libnvm*
cd /home/embed/Documents/GIDS/bam/build/bin

dmesg | grep "Character device" #check device name


cd /home/embed/Documents/GIDS/bam/build/module
sudo make unload
echo -n "0000:b3:00.0" > /sys/bus/pci/drivers/nvme/bind # 1T
echo -n "0000:1b:00.0" > /sys/bus/pci/drivers/nvme/bind # 7T

./nvm-readwrite_stripe-bench --input /home/embed/OGB/papers100M-bin/processed/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128

./nvm-readwrite_stripe-bench --input /mnt/nvme2n1/IGBH/full/processed/paper/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128

/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --input /home/embed/OGB/papers100M-bin/processed/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 --page_size 4096
/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --input /mnt/nvme2n1/IGBH/full/processed/paper/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 --page_size 4096

/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --input /home/embed/igb_datasets/small/processed/paper/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128
/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --input /home/embed/igb_datasets/small/processed/paper/re_node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128
/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --input /home/embed/igb_datasets/small/processed/paper/neighbor/neighbor_node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128

./nvm-readwrite_stripe-bench-libnvm1 
/home/embed/Documents/GIDS/bam/build/bin/nvm-readwrite_stripe-bench-libnvm1 --threads $((1024*1024*4)) --queue_depth 1024  --num_queues 128 --access_type 0 --input /mnt/nvme2n1/read_test.nvme

sudo ./nvm-readwrite_stripe-bench-libnvm1 --input /home/embed/Documents/GIDS/evaluation/first_part_edge_index.npy --queue_depth 1024 --access_type 0 --num_queues 128 --threads 1024 --n_ctrls 1 --ioffset 128 --page_size 4096

/home/embed/bam_gids/build/bin

./nvm-readwrite_stripe-bench-jc \
--input /home/embed/Downloads/igb_datasets/small/processed/paper/node_feat.npy \
--queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --ioffset 128 \
--page_size $((4*4096)) \
--libnvmName /dev/libnvm0