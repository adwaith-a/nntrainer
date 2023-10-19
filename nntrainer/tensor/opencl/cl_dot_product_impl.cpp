#include "cl_dot_product_impl.hpp"

#include "opencl_buffer.hpp"
#include "opencl_loader.hpp"

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {

template <typename T>
T *GpuCLDotProductImpl::CLEleMulImpl(T *matAdata, T *matBdata,
                                     std::vector<uint32_t> matAdims,
                                     std::vector<uint32_t> matBdims) {
  uint32_t output_size = matAdims[0] * matBdims[1];
  T *output_data = new T[output_size];

  bool result = false;

  do {
    result = Init();
    if (!result) {
      break;
    }

    size_t inputA_size = sizeof(T) * matAdims[0] * matAdims[1];
    Buffer inputA(context_inst_, inputA_size, true, nullptr);

    size_t inputB_size = sizeof(T) * matBdims[0] * matBdims[1];
    Buffer inputB(context_inst_, inputB_size, true, nullptr);

    Buffer output(context_inst_, sizeof(T) * output_size, false, nullptr);

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

    result = kernel_.SetKernelArguments(3, &matAdims[0], sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(4, &matAdims[1], sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(5, &matBdims[1], sizeof(int));
    if (!result) {
      break;
    }

    cl_event event;

    const int work_groups_count[3] = {(int)matAdims[0], (int)matBdims[1], 1};
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

template float *
GpuCLDotProductImpl::CLEleMulImpl<float>(float *matAdata, float *matBdata,
                                         std::vector<uint32_t> matAdims,
                                         std::vector<uint32_t> matBdims);

bool GpuCLDotProductImpl::Init() {
  // LOGI("Init");
  if (initialized_) {
    // LOGI("Already initialized");
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
    result =
      program.CreateCLProgram(context_inst_.GetContext(),
                              context_inst_.GetDeviceId(), mat_mul_kernel_, "");
    if (!result) {
      break;
    }

    result = kernel_.CreateKernelFromProgram(program, "mat_mul");
    if (!result) {
      break;
    }
    initialized_ = true;
  } while (false);

  return result;
}

GpuCLDotProductImpl::~GpuCLDotProductImpl() {
  if (initialized_) {
    command_queue_inst_.ReleaseCommandQueue();
    context_inst_.ReleaseContext();
  }
}

} // namespace nntrainer::internal