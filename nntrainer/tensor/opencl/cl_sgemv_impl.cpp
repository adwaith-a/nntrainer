#include "cl_sgemv_impl.hpp"
#include "opencl_buffer.hpp"
#include "opencl_loader.hpp"
#include <iostream>

namespace nntrainer::internal {

template <typename T>
T *GpuCLSgemvImpl::CLSgemvImpl(const T *matAdata, const T *vecXdata,
                               T *vecYdata, T alpha, T beta, unsigned int dim1,
                               unsigned int dim2) {

  std::cerr << "GpuCLSgemvImpl::CLSgemvImpl" << std::endl;
  //   T *output_data = new T[num_elems];

  bool result = false;

  do {
    result = Init();
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(T) * dim1;
    size_t dim2_size = sizeof(T) * dim2;
    Buffer inputA(context_inst_, dim1_size * dim2_size, true, nullptr);

    Buffer inputX(context_inst_, dim1_size, true, nullptr);

    Buffer inOutY(context_inst_, dim2_size, true, nullptr);

    result = inputA.WriteData(command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(3, &alpha, sizeof(T));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(4, &beta, sizeof(T));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(5, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(6, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, (int)dim2, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = command_queue_inst_.DispatchCommand(kernel_, work_groups_count,
                                                 work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);

  if (!result) {
    // LOGE("OpenCL Wrapper function failed");
  }

  return vecYdata;
}

// template _Float16 *GpuCLSgemvImpl::CLSgemvImpl<_Float16>(
//   const _Float16 *matAdata, const _Float16 *matBdata, const _Float16
//   *vecYdata, unsigned int dim1, unsigned int dim2);
template float *
GpuCLSgemvImpl::CLSgemvImpl<float>(const float *matAdata, const float *vecXdata,
                                   float *vecYdata, float alpha, float beta,
                                   unsigned int dim1, unsigned int dim2);

bool GpuCLSgemvImpl::Init() {
  if (initialized_) {
    return true;
  }
  bool result = false;

  do {
    result = command_queue_inst_.CreateCommandQueue();
    if (!result) {
      break;
    }

    Program program;
    result =
      program.CreateCLProgram(context_inst_.GetContext(),
                              context_inst_.GetDeviceId(), sgemv_kernel_, "");
    if (!result) {
      break;
    }

    result = kernel_.CreateKernelFromProgram(program, "sgemv");
    if (!result) {
      break;
    }
    initialized_ = true;
  } while (false);

  return result;
}

GpuCLSgemvImpl::~GpuCLSgemvImpl() {
  if (initialized_) {
    command_queue_inst_.ReleaseCommandQueue();
    context_inst_.ReleaseContext();
  }
}

} // namespace nntrainer::internal