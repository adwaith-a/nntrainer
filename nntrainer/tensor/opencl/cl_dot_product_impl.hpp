#ifndef GPU_CL_DOT_PRODUCT_IMPL_HPP_
#define GPU_CL_DOT_PRODUCT_IMPL_HPP_

#include "cl_op_impl.hpp"

namespace nntrainer::internal {
class GpuCLDotProductImpl : GpuCLOpImpl {
  std::string mat_mul_kernel_ =
    R"(__kernel void mat_mul(const __global float* A, const __global float* B,
                      __global float* C, int numRowsA, int numColsA,
                      int numColsB) {
            const int row = get_global_id(0);
            const int col = get_global_id(1);
            float sum = 0.0f;
            for (int i = 0; i < numColsA; ++i) {
                sum += A[row * numColsA + i] * B[col + numColsB * i];
            }
            C[row * numColsB + col] = sum;
        })";

public:
  bool Init() override;

  template <typename T>
  T *CLEleMulImpl(T *matAdata, T *matBdata, std::vector<uint32_t> matAdims,
                  std::vector<uint32_t> matBdims);

  ~GpuCLDotProductImpl();
};
} // namespace nntrainer::internal

#endif // GPU_CL_DOT_PRODUCT_IMPL_HPP_