#include <buffer.h>
#include <byteswap.h>
#include <ctrl.h>
#include <cuda.h>
#include <event.h>
#include <fcntl.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <page_cache.h>
#include <queue.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <util.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "settings.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;

// uint32_t n_ctrls = 1;
const char *const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3",
                                       "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
const char *const intel_ctrls_paths[] = {"/dev/libinvm0", "/dev/libinvm1", "/dev/libinvm9", "/dev/libinvm2", "/dev/libinvm3",
                                         "/dev/libinvm5", "/dev/libinvm6", "/dev/libinvm7", "/dev/libinvm8"};

#define SIZE (8 * 4096)

// feature type and dimension
typedef float FEATURE_TYPE;
#define FEATURE_DIM feature_dim

__global__ void print_cache_kernel(page_cache_d_t *pc) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        hexdump(pc->base_addr, SIZE);
    }
}

__global__ void new_kernel(ulonglong4 *dst, ulonglong4 *src, size_t num) { warp_memcpy<ulonglong4>(dst, src, num); }

int check_numpy_header(const char *file_path) {
    int fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "File cannot be opened\n");
        return -1;
    }
    char header[10];
    if (read(fd, header, 10) != 10) {
        fprintf(stderr, "Failed to read file header\n");
        close(fd);
        return -1;
    }
    close(fd);
    if (memcmp(header, "\x93NUMPY", 6) == 0) {
        int16_t header_len;
        memcpy(&header_len, header + 8, sizeof(int16_t));
        return 10 + header_len;
    }
    return 0;
}

template <typename T>
std::shared_ptr<T> mmap_file_with_offset(const char *file_path, std::size_t offset) {
    int fd = open(file_path, O_RDWR);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file");
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get file size");
    }

    size_t length = sb.st_size;

    void *mapped = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    void *map_in = (uint8_t *)mapped + offset;
    close(fd);

    if (mapped == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file");
    }

    auto deleter = [mapped, length](T *ptr) { munmap(mapped, length); };

    return std::shared_ptr<T>(static_cast<T *>(map_in), deleter);
}

template <typename T>
int file_element_count(const char *file_path, std::size_t offset) {
    struct stat sb;
    if (stat(file_path, &sb) == -1) {
        throw std::runtime_error("Failed to get file size");
    }
    return (sb.st_size - offset) / sizeof(T);
}

template <typename T>
std::shared_ptr<T> get_start_offset(std::shared_ptr<T> &each_size, int len) {
    uint64_t *start_offset_ptr = new uint64_t[len];
    std::shared_ptr<T> start_offset = std::shared_ptr<T>(start_offset_ptr, [](T *ptr) { delete[] ptr; });

    for (size_t i = 0; i < len; i++) {
        if (i == 0) {
            start_offset.get()[0] = 0;
            continue;
        }
        int prev_count = each_size.get()[i - 1];
        start_offset.get()[i] = start_offset.get()[i - 1] + prev_count;
    }
    return start_offset;
}

template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr, int64_t *index_ptr, int dim, int64_t num_idx, int cache_dim, uint64_t key_off) {
    uint64_t bid = blockIdx.x;
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int idx_idx = bid * num_warps + warp_id;
    if (idx_idx < num_idx) {  // 确保当前的全局索引不超出总索引范围
        bam_ptr<T> ptr(dr);

        uint64_t row_index = index_ptr[idx_idx] + key_off;
        uint64_t tid = threadIdx.x % 32;

        for (; tid < dim; tid += 32) {
            // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
            if (tid == 0) {
                // atomicAdd(d_gpu_access, 1);
            }
            T temp = ptr[(row_index)*cache_dim + tid];
            // printf("data: %llu\n",  (row_index) * cache_dim + tid);
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            // 将读取的数据存储到输出张量的对应位置
        }
    }
}

int main(int argc, char **argv) {
    Settings settings;
    try {
        settings.parseArguments(argc, argv);
    } catch (const string &e) {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {
        // 图节点重排
        bool graph_reorganize = settings.graph_reorganize;
        // 图节点重排的映射文件
        const char *graph_reorg_node_map_path;
        if (graph_reorganize) {
            if (settings.graph_reorg_node_map_path == nullptr) {
                fprintf(stderr, "Graph reorganization is enabled but no node map file is provided\n");
                return 1;
            }
            graph_reorg_node_map_path = settings.graph_reorg_node_map_path;
        }

        // 邻居节点放一起
        bool neighbor_feature = settings.neighbor_feature;
        // 邻居节点放一起后，全部节点的node_id
        const char *neighbor_feature_node_list_path;
        const char *neighbor_len_path;
        if (neighbor_feature) {
            if (settings.neighbor_feature_node_list_path == nullptr) {
                fprintf(stderr, "Neighbor feature is enabled but no node list file is provided\n");
                return 1;
            }
            neighbor_feature_node_list_path = settings.neighbor_feature_node_list_path;
            neighbor_len_path = settings.neighbor_len_path;
        }

        const char *node_sequence;
        const char *batch_size;
        node_sequence = settings.node_sequence;
        batch_size = settings.batch_size;

        int node_seq_offset = check_numpy_header(node_sequence);

        if (node_seq_offset == -1) {
            fprintf(stderr, "Failed to get npy header length\n");
            return 1;
        }

        std::cout << "Node sequence offset: " << node_seq_offset << std::endl;

        // 读取node sequence文件以及每一个batch的start_offset
        std::shared_ptr<uint64_t> map_node_sequence = mmap_file_with_offset<uint64_t>(node_sequence, node_seq_offset);
        int batch_size_offset = check_numpy_header(batch_size);
        if (batch_size_offset == -1) {
            fprintf(stderr, "Failed to get {Batch-Size} npy header length\n");
            exit(0);
        }
        std::cout << "Batch size offset: " << batch_size_offset << std::endl;
        int batch_len = file_element_count<uint64_t>(batch_size, batch_size_offset);
        std::cout << "Batch length(how many batch): " << batch_len << std::endl;
        std::shared_ptr<uint64_t> map_batch_size = mmap_file_with_offset<uint64_t>(batch_size, batch_size_offset);
        std::shared_ptr<uint64_t> batch_start_offset = get_start_offset<uint64_t>(map_batch_size, batch_len);

        // graph reorganization
        // 在一个page内有多个节点的feature，需要使用反向映射表快速找出某一个page内有哪些节点的feature；
        // uint64_t *reverse_map_graph_reorg_node;
        // 训练时使用的映射表
        std::shared_ptr<uint64_t> f_map_graph_reorg_node;
        if (graph_reorganize) {
            int offset = check_numpy_header(graph_reorg_node_map_path);
            if (offset == -1) {
                fprintf(stderr, "Failed to get {Graph Reorganization Node Map} npy header length\n");
                exit(0);
            }
            f_map_graph_reorg_node = mmap_file_with_offset<uint64_t>(graph_reorg_node_map_path, offset);
            // 需要知道映射文件中有多少个节点，才能算出来反向的映射表
            // int num_node = file_element_count<uint64_t>(graph_reorg_node_map_path, offset);
            // reverse_map_graph_reorg_node = new uint64_t[num_node];
            // for (size_t i = 0; i < num_node; i++) {
            //     reverse_map_graph_reorg_node[f_map_graph_reorg_node.get()[i]] = i;
            // }
        }

        // neighbor node 序列文件
        std::shared_ptr<uint64_t> f_neighbor_node_sequence;
        // 因为当前没有node都分散在哪些page里的这个信息，所以需要这个offset去寻找节点邻居列表的起始位置；每一个节点邻居列表的起始位置必然是其本身
        std::shared_ptr<uint64_t> f_nodes_neighbor_offset;
        if (neighbor_feature) {
            int offset = check_numpy_header(neighbor_feature_node_list_path);
            if (offset == -1) {
                fprintf(stderr, "Failed to get {Neighbor Feature Node List} npy header length\n");
                exit(0);
            }
            f_neighbor_node_sequence = mmap_file_with_offset<uint64_t>(neighbor_feature_node_list_path, offset);
            offset = check_numpy_header(neighbor_len_path);
            if (offset == -1) {
                fprintf(stderr, "Failed to get {Neighbor Len} npy header length\n");
                exit(0);
            }
            std::shared_ptr<uint64_t> f_nodes_neighbor_size = mmap_file_with_offset<uint64_t>(neighbor_len_path, offset);
            int num_nodes = file_element_count<uint64_t>(neighbor_len_path, offset);
            f_nodes_neighbor_offset = get_start_offset<uint64_t>(f_nodes_neighbor_size, num_nodes);
        }

        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        const char* ctrl_path = settings.libnvmName;
        std::vector<Controller *> ctrls(1);
        ctrls[0] = new Controller(ctrl_path, settings.nvmNamespace, settings.cudaDevice,
                                  settings.queueDepth, settings.numQueues);
        std::cout << "using ctrl: " << ctrl_path << std::endl;
        std::cout << "Ctrl page size: " << ctrls[0]->ctrl->page_size << std::endl;
        uint64_t page_size = settings.pageSize;
        uint64_t num_pages = settings.num_page;
        uint64_t numElems = num_pages * page_size / sizeof(FEATURE_TYPE);
        // 默认一个feature占据一个page,feature_dim和GPU一个warp内读取的数值个数有关系
        unsigned int feature_dim = page_size / sizeof(FEATURE_TYPE);
        // cache里能够承载的条目数量 (1024*1024 = MB)
        uint64_t cache_n_pages = settings.cache_size * 1024LL * 1024 / page_size;
        std::cout << "Cache npages: " << cache_n_pages << std::endl;

        uint32_t cudaDevice = 0;
        page_cache_t *h_pc = new page_cache_t(page_size, cache_n_pages, cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
        page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
        // bam的寻址空间
        printf("numElems: %lu\n", numElems);
        range_t<FEATURE_TYPE> *h_range =
            new range_t<FEATURE_TYPE>((uint64_t)0, numElems, (uint64_t)0, (uint64_t)num_pages, (uint64_t)0, (uint64_t)page_size, h_pc, cudaDevice,
                                      // REPLICATE
                                      STRIPE);

        range_d_t<FEATURE_TYPE> *d_range = (range_d_t<FEATURE_TYPE> *)h_range->d_range_ptr;
        std::vector<range_t<FEATURE_TYPE> *> vr;
        vr.push_back(h_range);
        array_t<FEATURE_TYPE> *a = new array_t<FEATURE_TYPE>(numElems, 0, vr, cudaDevice);

        double total_time = 0.0;
        uint64_t total_data = 0;
        int scale_factor = 1;
        for (size_t i = 3; i < (int)50; i++) {
            uint64_t node_count = 0;
            for (size_t j = 0; j < scale_factor; j++) {
                node_count += map_batch_size.get()[j * scale_factor + i];
            }
            uint64_t *index_ptr;
            uint64_t start_offset = batch_start_offset.get()[i * scale_factor];

            // node_sequence是当前要读取的node_id的序列
            uint64_t *node_sequence = new uint64_t[node_count];
            memcpy(node_sequence, map_node_sequence.get() + start_offset, node_count * sizeof(uint64_t));
            // TODO: 如果原始feature不是1024维度时，此处需要更改
            if (page_size == 4096) {
                // 每一个page只有一个feature
                // 4KB只测试raw和graph reorganization
                index_ptr = new uint64_t[node_count];
                if (graph_reorganize) {
                    for (size_t j = 0; j < node_count; j++) {
                        uint64_t node_id = node_sequence[j];
                        index_ptr[j] = f_map_graph_reorg_node.get()[node_id];
                    }
                } else {
                    memcpy(index_ptr, node_sequence, node_count * sizeof(uint64_t));
                }
            } else if (page_size == 16384) {
                // 每一个page有4个feature
                // 16KB测试raw、graph reorganization和neighbor feature
                // 当读过的16KB块中有节点的feature，则以后该节点的feature就不再读
                std::vector<uint64_t> page_sequence;
                if (graph_reorganize) {
                    // 图节点重排
                    std::unordered_set<uint64_t> page_set;
                    for (size_t j = 0; j < node_count; j++) {
                        uint64_t node_id = node_sequence[j];
                        int page_id = floor(f_map_graph_reorg_node.get()[node_id] / 4);
                        // if (page_set.find(page_id) != page_set.end()) {
                        //     continue;
                        // }
                        page_sequence.emplace_back(page_id);
                        page_set.insert(page_id);
                    }
                } else if (neighbor_feature) {
                    // 1跳邻居节点feature放一起
                    std::unordered_set<uint64_t> node_set;
                    for (size_t j = 0; j < node_count; j++) {
                        uint64_t node_id = node_sequence[j];
                        if (node_set.find(node_id) != node_set.end()) {
                            continue;
                        }
                        int idx = f_nodes_neighbor_offset.get()[node_id];
                        if (f_neighbor_node_sequence.get()[idx] != node_id) {
                            fprintf(stderr, "Neighbor node sequence map error\n");
                            return 1;
                        }
                        int page_id = floor(idx / 4);
                        for (size_t k = 0; k < 4; k++) {
                            node_set.insert(f_neighbor_node_sequence.get()[page_id * 4 + k]);
                        }
                        page_sequence.emplace_back(page_id);
                    }
                } else {
                    // 原始图
                    std::unordered_set<uint64_t> page_set;
                    for (size_t j = 0; j < node_count; j++) {
                        uint64_t node_id = node_sequence[j];
                        int page_id = floor(node_id / 4);
                        if (page_set.find(page_id) != page_set.end()) {
                            // 如果注释下面这一行，则无论之前的16KB页面是否被读取，都会再次读取
                            continue;
                        }
                        page_sequence.emplace_back(page_id);
                        page_set.insert(page_id);
                    }
                }
                node_count = page_sequence.size();
                index_ptr = new uint64_t[node_count];
                memcpy(index_ptr, page_sequence.data(), node_count * sizeof(uint64_t));
            } else {
                fprintf(stderr, "Unsupported page size\n");
                return 1;
            }
            delete[] node_sequence;

            uint64_t b_size = settings.blkSize;  // 128;
            // 每个块包含的warp数（一个warp通常包含32个线程）
            uint64_t n_warp = b_size / 32;
            // 计算所需的网格大小，确保每个索引都被处理
            uint64_t g_size = (node_count + n_warp - 1) / n_warp;

            // 为要取出来的feature在GPU上分配内存
            FEATURE_TYPE *d_data;
            cuda_err_chk(cudaMalloc(&d_data, node_count * FEATURE_DIM * sizeof(FEATURE_TYPE)));

            // 为索引在GPU上分配内存
            int64_t *d_index_ptr;
            printf("node_count: %lu\n", node_count);
            cuda_err_chk(cudaMalloc(&d_index_ptr, node_count * sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_index_ptr, index_ptr, node_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

            cuda_err_chk(cudaDeviceSynchronize());

            Event before;
            read_feature_kernel<FEATURE_TYPE><<<g_size, b_size>>>(a->d_array_ptr, d_data, d_index_ptr, FEATURE_DIM, node_count, FEATURE_DIM, 0);
            Event after;
            cuda_err_chk(cudaDeviceSynchronize());

            double elapsed = after - before;
            total_time += elapsed;
            uint64_t data = node_count * FEATURE_DIM * sizeof(FEATURE_TYPE);
            total_data += data;

            cudaFree(d_data);
            cudaFree(d_index_ptr);
            std::cout << "Batch " << i << " done" << std::endl;
        }
        simt::atomic<uint64_t, simt::thread_scope_device> access_counter;
        cuda_err_chk(cudaMemcpy(&access_counter, ctrls[0]->d_ctrl_ptr, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
        uint64_t ssd_access_count = access_counter.load();
        double ssd_data_size = ssd_access_count * 16 * 1024;
        double ssd_bandwidth = (ssd_data_size / (total_time / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
        a->print_reset_stats();
        ctrls[0]->print_reset_stats();
        double bandwidth = (((double)total_data) / (total_time / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
        // std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops:
        // "<< ios << "\tData Size (bytes): " << data << std::endl; std::cout <<
        // std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " <<
        // bandwidth << std::endl;
        printf("Elapsed Time(s): %.6f\n", total_time / (1000.0 * 1000.0));
        printf("SSD Data Size (GBytes): %.6f\n", ssd_data_size / (1024.0 * 1024.0 * 1024.0));
        printf("Total Data Size (GBytes): %.6f\n", total_data / 1024.0 / 1024.0 / 1024.0);
        printf("SSD Bandwidth (GB/S): %.6f\n", ssd_bandwidth);
        printf("Effective Bandwidth(GB/S): %.6f\n", bandwidth);
    } catch (const error &e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }
}
