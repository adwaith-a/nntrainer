#include "cl_add_impl.hpp"

#include "opencl_buffer.hpp"
#include "opencl_loader.hpp"
#include <iostream>

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {

template <typename T>
T *GpuCLAddImpl::CLEleAddImpl(const T *matAdata, const T *matBdata,
                              int num_elems) {
  std::cerr << "GpuCLAddImpl::CLEleAddImpl" << std::endl;
  T *output_data = new T[num_elems];

  bool result = false;

  do {
    result = Init();
    if (!result) {
      break;
    }

    size_t input_size = sizeof(T) * num_elems;
    Buffer inputA(context_inst_, input_size, true, nullptr);

    Buffer inputB(context_inst_, input_size, true, nullptr);

    Buffer output(context_inst_, input_size, false, nullptr);

    result = inputA.WriteData(command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputB.WriteData(command_queue_inst_, matBdata);
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(2, &output, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(3, &num_elems, sizeof(int));
    if (!result) {
      break;
    }

    cl_event event;

    const int work_groups_count[3] = {(int)num_elems, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = command_queue_inst_.DispatchCommand(kernel_, work_groups_count,
                                                 work_group_size);
    if (!result) {
      break;
    }

    result = output.ReadData(command_queue_inst_, output_data);
    if (!result) {
      break;
    }

    // cl_int err;
    // cl_ulong start_time;
    // cl_ulong end_time;
    // err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
    //                               sizeof(cl_ulong), &start_time, NULL);
    // err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
    //                               sizeof(cl_ulong), &end_time, NULL);

    // double elapsed_time = (end_time - start_time) * 1.0e-6;

    // LOGI("Opencl MatMul time taken: %f", elapsed_time);

  } while (false);

  if (!result) {
    // LOGE("OpenCL Wrapper function failed");
  }

  return output_data;
}

template float *GpuCLAddImpl::CLEleAddImpl<float>(const float *matAdata,
                                                  const float *matBdata,
                                                  int num_elems);

bool GpuCLAddImpl::Init() {
  // LOGI("Init");
  if (initialized_) {
    // // LOGI("Already initialized");
    return true;
  }
  // LOGI("Initializing");
  bool result = false;

  do {
    result = command_queue_inst_.CreateCommandQueue();
    if (!result) {
      break;
    }

    Program program;
    result = program.CreateCLProgram(
      context_inst_.GetContext(), context_inst_.GetDeviceId(), add_kernel_, "");
    if (!result) {
      break;
    }

    result = kernel_.CreateKernelFromProgram(program, "add");
    if (!result) {
      break;
    }
    initialized_ = true;
  } while (false);

  return result;
}

GpuCLAddImpl::~GpuCLAddImpl() {
  if (initialized_) {
    command_queue_inst_.ReleaseCommandQueue();
    context_inst_.ReleaseContext();
  }
}

} // namespace nntrainer::internal