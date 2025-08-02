#ifndef BAMNVME_H
#define BAMNVME_H

#include <buffer.h>
#include <cuda.h>
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
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <util.h>

#include <ctrl.h>
#include <event.h>
#include <page_cache.h>
#include <queue.h>

//#define TYPE float
struct GIDS_Controllers {
  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
  std::vector<Controller *> ctrls;

  uint32_t n_ctrls = 1;
  uint64_t queueDepth = 1024;
  uint64_t numQueues = 128;
  
  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  
  //member functions
  void init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,  const std::vector<int>& ssd_list);

};

template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;
    TYPE* device_cpu_buffer;
    uint64_t cpu_buffer_dim;
    uint64_t cpu_buffer_len;
};


/**
 * @brief The BAM_Feature_Store class manages features stored in NVMe storage with CUDA integration.
 *
 * This class provides functionality to initialize NVMe controllers, manage caching mechanisms,
 * read and write features, and interface with CUDA device memory. It supports different data types
 * through templating.
 *
 * @tparam TYPE The data type of the features (e.g., float, double).
 */
template <typename TYPE>
struct BAM_Feature_Store {
    /**
     * @brief Structure holding CPU buffer information for caching frequently accessed data.
     */
    GIDS_CPU_buffer<TYPE> CPU_buffer;

    bool cpu_buffer_flag = false;     ///< Indicates if the CPU buffer is initialized.
    bool seq_flag = true;             ///< Indicates if sequential access mode is enabled.

    uint64_t* offset_array;           ///< Array of offsets used for sampling.

    int dim;                          ///< Dimensionality of the feature vectors.
    uint64_t total_access;            ///< Total number of feature access operations.
    unsigned long long int cpu_access_count = 0;///< Counter for the number of CPU buffer accesses.
    unsigned long long int* d_cpu_access;       ///< Device pointer to the CPU access counter.

    unsigned long long int cpu_access_count_total = 0;

    unsigned long long int gpu_access_count = 0;
    unsigned long long int* d_gpu_access;
    unsigned long long int* d_cpu_clock;
    unsigned long long int* d_gpu_clock;

    unsigned long long int gpu_access_count_total = 0;
    // GPU去访问数据花费的时间
    double data_access_time = 0;

    //CPU访问时间
    unsigned long long int cpu_access_time = 0;
    unsigned long long int gpu_access_time = 0;
    unsigned long long int cpu_access_time_total = 0;
    unsigned long long int gpu_access_time_total = 0;

    // BAM parameters
    uint32_t cudaDevice = 0;          ///< CUDA device ID.
    size_t numPages = 262144 * 8;     ///< Number of pages in the cache.
    bool stats = false;               ///< Flag to enable statistics collection.
    size_t numThreads = 64;           ///< Number of threads per CUDA block.
    uint32_t domain = 0;              ///< PCI domain number (unused).
    uint32_t bus = 0;                 ///< PCI bus number (unused).
    uint32_t devfn = 0;               ///< PCI device/function number (unused).

    uint32_t n_ctrls = 1;             ///< Number of NVMe controllers.
    size_t blkSize = 128;             ///< Block size for CUDA kernels.
    size_t queueDepth = 1024;         ///< Depth of NVMe queues.
    size_t numQueues = 128;           ///< Number of queues per NVMe controller.
    uint32_t pageSize = 4096;         ///< Page size used in NVMe storage.
    uint64_t numElems = 300LL * 1000 * 1000 * 1024; ///< Total number of elements/features.
    uint64_t read_offset = 0;         ///< Offset for reading data from storage.
    std::vector<Controller*> ctrls;   ///< Vector of pointers to NVMe controllers.

    page_cache_t* h_pc;               ///< Host-side page cache pointer.
    range_t<TYPE>* h_range;           ///< Host-side range representing data.
    std::vector<range_t<TYPE>*> vr;   ///< Vector of range pointers.
    array_t<TYPE>* a;                 ///< Array managing device memory.
    range_d_t<TYPE>* d_range;         ///< Device-side range pointer.

    float kernel_time = 0;            ///< Accumulated time spent in CUDA kernels.

    /**
     * @brief Initializes the feature store with NVMe controllers and caching parameters.
     *
     * @param GIDS_ctrl   The GIDS_Controllers object containing NVMe controller information.
     * @param ps          Page size.
     * @param r_off       Read offset.
     * @param num_ele     Number of elements/features to manage.
     * @param cache_size  Size of the cache in pages or megabytes.
     * @param num_ssd     Number of SSDs to utilize.
     */
    void init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size,
                          uint64_t num_ssd);

    /**
     * @brief Reads features from NVMe storage into a tensor.
     *
     * @param tensor_ptr  Device pointer where the features will be stored.
     * @param index_ptr   Device pointer containing indices of features to read.
     * @param num_index   Number of indices/features to read.
     * @param dim         Dimensionality of the features.
     * @param cache_dim   Dimensionality used for caching mechanisms.
     * @param key_off     Offset applied to feature indices.
     */
    void read_feature(uint64_t tensor_ptr, uint64_t index_ptr, int64_t num_index, int dim, int cache_dim, uint64_t key_off);

    void get_io_stat(uint64_t tensor_ptr);
    /**
     * @brief Reads features using multiple iterations with heterogeneous parameters.
     *
     * @param num_iter            Number of iterations.
     * @param i_ptr_list          List of device pointers for tensors in each iteration.
     * @param i_index_ptr_list    List of device pointers for indices in each iteration.
     * @param num_index           List containing the number of indices for each iteration.
     * @param dim                 Dimensionality of the features.
     * @param cache_dim           Dimensionality used for caching mechanisms.
     * @param key_off             List of offsets applied to indices in each iteration.
     */
    void read_feature_hetero(int num_iter, const std::vector<uint64_t>& i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list,
                             const std::vector<uint64_t>& num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);

    /**
     * @brief Merged feature reading operation for multiple iterations.
     *
     * @param num_iter            Number of iterations.
     * @param i_ptr_list          List of device pointers for tensors in each iteration.
     * @param i_index_ptr_list    List of device pointers for indices in each iteration.
     * @param num_index           List containing the number of indices for each iteration.
     * @param dim                 Dimensionality of the features.
     * @param cache_dim           Dimensionality used for caching mechanisms.
     */
    void read_feature_merged(int num_iter, const std::vector<uint64_t>& i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list,
                             const std::vector<uint64_t>& num_index, int dim, int cache_dim);

    /**
     * @brief Merged feature reading with heterogeneous parameters for multiple iterations.
     *
     * @param num_iter            Number of iterations.
     * @param i_ptr_list          List of device pointers for tensors in each iteration.
     * @param i_index_ptr_list    List of device pointers for indices in each iteration.
     * @param num_index           List containing the number of indices for each iteration.
     * @param dim                 Dimensionality of the features.
     * @param cache_dim           Dimensionality used for caching mechanisms.
     * @param key_off             List of offsets applied to indices in each iteration.
     */
    void read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>& i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list,
                                    const std::vector<uint64_t>& num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);

    /**
     * @brief Initializes a CPU backing buffer for caching.
     *
     * @param dim  Dimensionality of the features.
     * @param len  Number of features to store in the CPU buffer.
     */
    void cpu_backing_buffer(uint64_t dim, uint64_t len);

    /**
     * @brief Sets up the CPU buffer with specified indices.
     *
     * @param idx_buffer  Device pointer containing indices.
     * @param num         Number of indices to set in the CPU buffer.
     */
    void set_cpu_buffer(uint64_t idx_buffer, int num);

    /**
     * @brief Configures window buffering parameters for feature retrieval.
     *
     * @param id_idx     Device pointer to the index buffer.
     * @param num_pages  Number of pages to buffer.
     * @param hash_off   Hash offset applied to indices.
     */
    void set_window_buffering(uint64_t id_idx, int64_t num_pages, int hash_off);

    /**
     * @brief Prints and resets statistical data for the feature store and NVMe controllers.
     */
    void print_stats();

    /**
     * @brief Prints and resets statistical data for the feature store without controller details.
     */
    void print_stats_no_ctrl();

    /**
     * @brief Retrieves the device pointer to the array managing device memory.
     *
     * @return Device pointer to the underlying array object.
     */
    uint64_t get_array_ptr();

    /**
     * @brief Retrieves the offset array used for sampling operations.
     *
     * @return Device pointer to the offset array.
     */
    uint64_t get_offset_array();

    /**
     * @brief Sets offsets for input, index, and data during operations.
     *
     * @param in_off     Input offset.
     * @param index_off  Index offset.
     * @param data_off   Data offset.
     */
    void set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off);

    /**
     * @brief Stores a tensor to NVMe storage at a specified offset.
     *
     * @param tensor_ptr  Device pointer to the tensor data.
     * @param num         Number of elements to store.
     * @param offset      Offset in storage where the tensor will be written.
     */
    void store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset);

    /**
     * @brief Reads a tensor from NVMe storage starting at a specified offset.
     *
     * @param num     Number of elements to read.
     * @param offset  Offset in storage from where to start reading.
     */
    void read_tensor(uint64_t num, uint64_t offset);

    /**
     * @brief Flushes the cache, ensuring all pending operations are completed.
     */
    void flush_cache();

    /**
     * @brief Retrieves the current CPU access count.
     *
     * @return Number of times the CPU buffer was accessed.
     */
    unsigned long long int get_cpu_access_count();

    /**
     * @brief Resets the CPU access count to zero.
     */
    void flush_cpu_access_count();
};

#endif
