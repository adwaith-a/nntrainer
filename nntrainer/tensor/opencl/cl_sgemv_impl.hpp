#ifndef GPU_CL_SGEMV_IMPL_HPP_
#define GPU_CL_SGEMV_IMPL_HPP_

#include "cl_op_impl.hpp"

namespace nntrainer::internal {
class GpuCLSgemvImpl : public GpuCLOpImpl {
  std::string sgemv_kernel_ =
    R"(__kernel void sgemv(const __global float* A, const __global float* X,
                      __global float* Y, unsigned int M, unsigned int N, float alpha, float beta) {
        const int row = get_global_id(0);
        Y[row] = Y[row] * beta;
        for (unsigned int j = 0; j < N; j++){
            Y[row] += alpha * A[row * N + j] * X[j];
        }
    })";

public:
  bool Init() override;

  template <typename T>
  T *CLSgemvImpl(const T *matAdata, const T *vecXdata, T *vecYdata, T alpha,
                 T beta, unsigned int dim1, unsigned int dim2);

  ~GpuCLSgemvImpl();
};
} // namespace nntrainer::internal

#endif // GPU_CL_SGEMV_IMPL_HPP_