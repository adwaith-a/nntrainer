#ifndef GPU_CL_ADD_IMPL_HPP_
#define GPU_CL_ADD_IMPL_HPP_

#include "cl_op_impl.hpp"
namespace nntrainer::internal {
class GpuCLAddImpl : public GpuCLOpImpl {
  std::string add_kernel_ =
    R"(__kernel void add(const __global float* A, const __global float* B,
                      __global float* C, int num_elems) {
            const int idx = get_global_id(0);
            C[idx] = A[idx] + B[idx];
        })";

public:
  bool Init() override;

  template <typename T>
  T *CLEleAddImpl(const T *matAdata, const T *matBdata, int num_elems);

  ~GpuCLAddImpl();
};
} // namespace nntrainer::internal

#endif // GPU_CL_ADD_IMPL_HPP_