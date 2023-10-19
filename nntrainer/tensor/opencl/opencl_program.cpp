#include "opencl_program.hpp"

#include <string>

#include "opencl_loader.hpp"

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {

bool Program::BuildProgram(cl_device_id device_id,
                           const std::string &compiler_options) {
  const int error_code = clBuildProgram(
    program_, 0, nullptr, compiler_options.c_str(), nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to build program executable. OpenCL error code: %d. %s",
    //  error_code,
    //  (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  }

  return true;
}

std::string Program::GetProgramBuildInfo(cl_device_id device_id,
                                         cl_program_build_info info) {
  size_t size;
  cl_int error_code =
    clGetProgramBuildInfo(program_, device_id, info, 0, nullptr, &size);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to GetProgramBuildInfo. OpenCL error code: %d", error_code);
    return "";
  }

  std::string result(size - 1, 0);
  error_code =
    clGetProgramBuildInfo(program_, device_id, info, size, &result[0], nullptr);
  if (error_code != CL_SUCCESS) {
    // LOGE("Failed to GetProgramBuildInfo. OpenCL error code: %d", error_code);
    return "";
  }
  return result;
}

bool Program::CreateCLProgram(const cl_context &context,
                              const cl_device_id &device_id,
                              const std::string &code,
                              const std::string &compiler_options) {
  int error_code;
  const char *source = code.c_str();

  program_ =
    clCreateProgramWithSource(context, 1, &source, nullptr, &error_code);
  if (!program_ || error_code != CL_SUCCESS) {
    // LOGE("Failed to create compute program. OpenCL error code: %d",
    // error_code);
    return false;
  }

  return BuildProgram(device_id, compiler_options);
}

const cl_program &Program::GetProgram() { return program_; }

} // namespace nntrainer::internal