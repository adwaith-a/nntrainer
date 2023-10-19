#ifndef GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_
#define GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_

#include "cl.h"
#include "opencl_kernel.hpp"

namespace nntrainer::internal {
class CommandQueueManager {
  cl_command_queue command_queue_{nullptr};

  CommandQueueManager(){};

public:
  static CommandQueueManager &GetInstance();
  bool CreateCommandQueue();
  void ReleaseCommandQueue();

  bool EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes, void *data,
                         bool async = false);
  bool EnqueueWriteBuffer(cl_mem buffer, size_t size_in_bytes, const void *data,
                          bool async = false);

  bool DispatchCommand(Kernel kernel, const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr);

  const cl_command_queue GetCommandQueue();

  void operator=(CommandQueueManager const &) = delete;
  CommandQueueManager(CommandQueueManager const &) = delete;
  ~CommandQueueManager();
};
} // namespace nntrainer::internal

#endif // GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_