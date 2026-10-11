#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Keep tensor/tensor FP32 division, including subnormals, rather than replacing
// it by multiplication by an approximate reciprocal. No fast-math build flags.
template <typename Degree, bool Backward>
__global__ void mean_normalize_kernel(const float* x, const Degree* degree,
    float* y, int64_t rows, int64_t columns, int64_t stride0, int64_t stride1,
    int64_t degree_stride, int64_t maximum) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < rows * columns; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = i / columns;
    const int64_t col = i % columns;
    int64_t d = static_cast<int64_t>(degree[row * degree_stride]);
    d = d < 1 ? 1 : (d > maximum ? maximum : d);
    y[i] = __fdiv_rn(x[row * stride0 + col * stride1], static_cast<float>(d));
  }
}

template <bool Backward>
void launch(uintptr_t xp, uintptr_t dp, uintptr_t yp, int64_t rows,
    int64_t columns, int64_t stride0, int64_t stride1, int64_t degree_stride,
    int64_t maximum, int bits, uintptr_t stream_pointer) {
  if (rows < 0 || columns < 0 || stride0 < 0 || stride1 < 0 || degree_stride < 0 ||
      maximum < 1 || (bits != 32 && bits != 64) ||
      (columns && rows > std::numeric_limits<int64_t>::max() / columns))
    throw std::invalid_argument("Invalid mean normalization shape/degree contract");
  if (!rows || !columns) return;
  if (!xp || !dp || !yp) throw std::invalid_argument("Null nonempty tensor");
  const int64_t count = rows * columns;
  const int blocks = static_cast<int>((count - 1) / 256 + 1 > 65535 ? 65535 : (count - 1) / 256 + 1);
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (bits == 32)
    mean_normalize_kernel<int32_t, Backward><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(xp), reinterpret_cast<const int32_t*>(dp),
      reinterpret_cast<float*>(yp), rows, columns, stride0, stride1, degree_stride, maximum);
  else
    mean_normalize_kernel<int64_t, Backward><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const float*>(xp), reinterpret_cast<const int64_t*>(dp),
      reinterpret_cast<float*>(yp), rows, columns, stride0, stride1, degree_stride, maximum);
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(std::string("mean normalization: ") + cudaGetErrorString(error));
}

PYBIND11_MODULE(DiGiTMeanNormalizeCUDA, m) {
  m.attr("MEAN_NORMALIZE_API") = 1;
  m.def("forward", &launch<false>);
  m.def("backward", &launch<true>);
}
