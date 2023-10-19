#ifndef GPU_CL_OPENCL_BUFFER_HPP_
#define GPU_CL_OPENCL_BUFFER_HPP_

#include "cl.h"
#include "opencl_command_queue_manager.hpp"
#include "opencl_context_manager.hpp"

namespace nntrainer::internal {
class Buffer {
  cl_mem mem_buf_{nullptr};
  size_t size_{0};
  void Release();

public:
  Buffer(){};
  Buffer(ContextManager &context_manager, int size_in_bytes, bool read_only,
         void *data);

  // Move only
  Buffer(Buffer &&buffer);
  Buffer &operator=(Buffer &&buffer);
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  ~Buffer();
  cl_mem &GetBuffer();

  bool WriteData(CommandQueueManager &command_queue_inst, const void *data);
  bool ReadData(CommandQueueManager &command_queue_inst, void *data);
};
} // namespace nntrainer::internal
#endif // GPU_CL_OPENCL_BUFFER_HPP_