#include "opencl_command_queue_manager.hpp"

#include "opencl_context_manager.hpp"
#include "opencl_loader.hpp"

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {
CommandQueueManager &CommandQueueManager::GetInstance() {
  static CommandQueueManager instance;
  return instance;
}

bool CommandQueueManager::CreateCommandQueue() {
  if (command_queue_) {
    clRetainCommandQueue(command_queue_);
    return true;
  }

  int error_code;
  ContextManager &context_instance = ContextManager::GetInstance();
  cl_context context = context_instance.GetContext();
  cl_device_id device_id = context_instance.GetDeviceId();
  command_queue_ = clCreateCommandQueue(context, device_id, 0, &error_code);
  if (!command_queue_) {
    // LOGE("Failed to create a command queue. OpenCL error code: %d",
    // error_code);
    return false;
  }
  clRetainCommandQueue(command_queue_);
  return true;
}

void CommandQueueManager::ReleaseCommandQueue() {
  if (command_queue_) {
    clReleaseCommandQueue(command_queue_);
  }
}

CommandQueueManager::~CommandQueueManager() {
  if (command_queue_) {
    clReleaseCommandQueue(command_queue_);
    command_queue_ = nullptr;
    ContextManager::GetInstance().ReleaseContext();
  }
}

const cl_command_queue CommandQueueManager::GetCommandQueue() {
  return command_queue_;
}

bool CommandQueueManager::EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes,
                                            void *data, bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code =
    clEnqueueReadBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                        data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to read data from GPU (clEnqueueReadBuffer). OpenCL error "
    //  "code: %d",
    //  error_code);
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueWriteBuffer(cl_mem buffer,
                                             size_t size_in_bytes,
                                             const void *data, bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code =
    clEnqueueWriteBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                         data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to upload data to GPU (clEnqueueWriteBuffer). OpenCL error "
    //  "code: %d",
    //  error_code);
    return false;
  }

  return true;
}

bool CommandQueueManager::DispatchCommand(Kernel kernel,
                                          const int (&work_groups_count)[3],
                                          const int (&work_group_size)[3],
                                          cl_event *event) {
  // std::array<size_t, 3> local;
  // std::array<size_t, 3> global;
  //  for (int i = 0; i < 3; ++i) {
  //    local[i] = work_group_size[i];
  //    global[i] = work_groups_count[i] * work_group_size[i];
  //  }

  const int TS = 32;
  const size_t local[2] = {TS, TS};
  const size_t global[2] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1])};
  cl_kernel kernel_ = kernel.GetKernel();
  const int error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel_, 2, nullptr, global, local, 0, nullptr, event);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d",
    // error_code);
    return false;
  }

  return true;
}

} // namespace nntrainer::internal
