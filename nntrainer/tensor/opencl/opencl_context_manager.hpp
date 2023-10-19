#ifndef GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_
#define GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_

#include <mutex>

#include "cl.h"

namespace nntrainer::internal {
class ContextManager {
  cl_platform_id platform_id_{nullptr};
  cl_device_id device_id_{nullptr};
  cl_context context_{nullptr};

  bool CreateDefaultGPUDevice();
  bool CreateCLContext();

  ContextManager(){};

public:
  static ContextManager &GetInstance();

  const cl_context &GetContext();
  void ReleaseContext();

  const cl_device_id GetDeviceId();

  void operator=(ContextManager const &) = delete;
  ContextManager(ContextManager const &) = delete;
  ~ContextManager();
};
} // namespace nntrainer::internal
#endif // GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_