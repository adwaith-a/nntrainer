#include "opencl_kernel.hpp"

#include "opencl_loader.hpp"

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {

bool Kernel::CreateKernelFromProgram(Program program,
                                     const std::string &function_name) {
  int error_code;
  cl_program prgm = program.GetProgram();
  kernel_ = clCreateKernel(prgm, function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    // LOGE("Failed to create %s. OpenCL error code: %d", function_name.c_str(),
    //     error_code);
    return false;
  }
  clRetainProgram(prgm);

  return true;
}

bool Kernel::SetKernelArguments(cl_uint arg_index, const void *arg_value,
                                size_t size) {
  int error_code;
  error_code = clSetKernelArg(kernel_, arg_index, size, arg_value);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to set argument. OpenCL error code: %d", error_code);
    return false;
  }

  return true;
}

const cl_kernel Kernel::GetKernel() { return kernel_; }

} // namespace nntrainer::internal