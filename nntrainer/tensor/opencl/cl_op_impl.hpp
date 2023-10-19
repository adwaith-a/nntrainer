#ifndef GPU_CL_OP_IMPL_HPP_
#define GPU_CL_OP_IMPL_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "opencl_command_queue_manager.hpp"
#include "opencl_context_manager.hpp"
#include "opencl_kernel.hpp"
#include "opencl_program.hpp"

namespace nntrainer::internal {
class GpuCLOpImpl {
protected:
  bool initialized_;
  Kernel kernel_;
  ContextManager &context_inst_ = ContextManager::GetInstance();
  CommandQueueManager &command_queue_inst_ = CommandQueueManager::GetInstance();

public:
  virtual bool Init() = 0;
};
} // namespace nntrainer::internal

#endif // GPU_CL_OP_IMPL_HPP_