#ifndef GPU_CL_OPENCL_KERNEL_HPP_
#define GPU_CL_OPENCL_KERNEL_HPP_

#include <string>

#include "cl.h"
#include "opencl_program.hpp"

namespace nntrainer::internal {
class Kernel {
  cl_kernel kernel_{nullptr};

public:
  bool CreateKernelFromProgram(Program program,
                               const std::string &function_name);

  bool SetKernelArguments(cl_uint arg_index, const void *arg_value,
                          size_t size);
  const cl_kernel GetKernel();
};
} // namespace nntrainer::internal
#endif // GPU_CL_OPENCL_KERNEL_HPP_