#ifndef GPU_CL_OPENCL_PROGRAM_HPP_
#define GPU_CL_OPENCL_PROGRAM_HPP_

#include <string>

#include "cl.h"

namespace nntrainer::internal {
class Program {
  cl_program program_{nullptr};

  bool BuildProgram(cl_device_id device_id,
                    const std::string &compiler_options);
  std::string GetProgramBuildInfo(cl_device_id device_id,
                                  cl_program_build_info info);

public:
  bool CreateCLProgram(const cl_context &context, const cl_device_id &device_id,
                       const std::string &code,
                       const std::string &compiler_options);
  const cl_program &GetProgram();
};
} // namespace nntrainer::internal
#endif // GPU_CL_OPENCL_PROGRAM_HPP_