#include "opencl_loader.hpp"

#include <dlfcn.h>

#include <string>

// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
// __VA_ARGS__) #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
// __VA_ARGS__)

namespace nntrainer::internal {

#define LoadFunction(function) \
  function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));

void LoadOpenCLFunctions(void *libopencl);

static bool open_cl_initialized = false;

bool LoadOpenCL() {
  if (open_cl_initialized) {
    return true;
  }

  void *libopencl = nullptr;
  static const char *kClLibName = "libOpenCL.so";

  libopencl = dlopen(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    open_cl_initialized = true;
    return true;
  }

  // record error
  std::string error(dlerror());
  // LOGE("Can not open OpenCL library on this device - %s", error.c_str());
  return false;
}

void LoadOpenCLFunctions(void *libopencl) {
  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clBuildProgram);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clRetainProgram);
  LoadFunction(clCreateKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadFunction(clReleaseMemObject);
}

PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clBuildProgram clBuildProgram;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clRetainProgram clRetainProgram;
PFN_clCreateKernel clCreateKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clReleaseMemObject clReleaseMemObject;

} // namespace nntrainer::internal
