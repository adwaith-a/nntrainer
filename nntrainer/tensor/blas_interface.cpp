// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_interface.cpp
 * @date   28 Aug 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is dummy header for blas support
 *
 */

#include <blas_interface.h>
#include <blas_neon.h>
#include <cmath>
#include <iostream>
#include <nntrainer_error.h>

#define sgemv_loop(ci, cj, cM, cN)           \
  do {                                       \
    double y0;                               \
    unsigned int i, j;                       \
    for (ci = 0; ci != cM; ci++) {           \
      y0 = Y[ci * incy] * beta;              \
      for (cj = 0; cj != cN; cj++)           \
        y0 += A[i + j * lda] * X[cj * incx]; \
      Y[ci * incy] = y0;                     \
    }                                        \
  } while (0);

namespace nntrainer {

#ifdef ENABLE_FP16
static void saxpy_FP16(const unsigned int N, const float alpha, const __fp16 *X,
                       const int incX, __fp16 *Y, const int incY) {
  if (incX < 0 or incY < 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + alpha * X[i * incX];
}

static void sgemv_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const __fp16 *A,
                       const unsigned int lda, const __fp16 *X, const int incX,
                       const float beta, __fp16 *Y, const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  std::cout << "rows : " << M << " , cols : " << N << std::endl;

  if (TransA == CblasTrans) {
    if (incX == 1 && incY == 1 && (N % 16 == 0 || N % 8 == 0 || N % 4 == 0)) {
      std::cout << "Using neon_transpose" << std::endl;
      nntrainer::neon::sgemv_transpose_neon_fp16(A, X, Y, M, N, alpha, beta);
      // sgemv_loop(i, j, N, M);

    } else {
      std::cout << "Using sgemv_loop" << std::endl;
      sgemv_loop(i, j, N, M);
    }
  } else {
    if (incX == 1 && incY == 1 && (N % 16 == 0 || N % 8 == 0 || N % 4 == 0)) {
      std::cout << "Using neon" << std::endl;
      nntrainer::neon::sgemv_neon_fp16(A, X, Y, M, N, alpha, beta);

    } else {
      std::cout << "Using sgemv_loop_transpose" << std::endl;
      sgemv_loop(j, i, M, N);
    }
  }
  // if (TransA == CblasTrans) {

  //   sgemv_loop(i, j, N, M);
  // } else {

  //   sgemv_loop(j, i, M, N);
  // }
}

static void sgemv_FP16_loop(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                            const unsigned int M, const unsigned int N,
                            const float alpha, const __fp16 *A,
                            const unsigned int lda, const __fp16 *X,
                            const int incX, const float beta, __fp16 *Y,
                            const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  std::cout << "rows : " << M << " , cols : " << N << std::endl;

  if (TransA == CblasTrans) {
    if (incX == 1 && incY == 1 && (N % 16 == 0 || N % 8 == 0 || N % 4 == 0)) {
      sgemv_loop(i, j, N, M);

    } else {
      sgemv_loop(i, j, N, M);
    }
  } else {
    if (incX == 1 && incY == 1 && (N % 16 == 0 || N % 8 == 0 || N % 4 == 0)) {
      sgemv_loop(j, i, M, N);

    } else {
      sgemv_loop(j, i, M, N);
    }
  }
  // if (TransA == CblasTrans) {

  //   sgemv_loop(i, j, N, M);
  // } else {

  //   sgemv_loop(j, i, M, N);
  // }
}

static __fp16 sdot_FP16(const unsigned int N, const __fp16 *X,
                        const unsigned int incX, const __fp16 *Y,
                        const unsigned int incY) {
  __fp16 ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

static void scopy_FP16(const unsigned int N, const __fp16 *X, const int incX,
                       __fp16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

void sscal(const unsigned int N, const float alpha, __fp16 *X, const int incX) {
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = alpha * X[i * incx];
}

static float snrm2_FP16(const unsigned int N, const __fp16 *X, const int incX) {
  unsigned int incx = abs(incX);
  __fp16 sum = 0.0f;
  __fp16 tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}
static void sgemm_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       CBLAS_TRANSPOSE TransB, const unsigned int M,
                       const unsigned int N, const unsigned int K,
                       const float alpha, const __fp16 *A,
                       const unsigned int lda, const __fp16 *B,
                       const unsigned int ldb, const float beta, __fp16 *C,
                       const unsigned int ldc) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      __fp16 c = 0.0;
      __fp16 c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        __fp16 a, b;
        a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
    }
  }
}

static unsigned int isamax_FP16(const unsigned int N, const __fp16 *X,
                                const int incX) {

  unsigned int max_idx = 0;
  __fp16 max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    __fp16 cur_val = abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

void saxpy(const unsigned int N, const float alpha, const __fp16 *X,
           const int incX, __fp16 *Y, const int incY) {
  saxpy_FP16(N, alpha, X, incX, Y, incY);
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const __fp16 *A, const unsigned int lda,
           const __fp16 *B, const unsigned int ldb, const float beta, __fp16 *C,
           const unsigned int ldc) {
  sgemm_FP16(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
             ldc);
}

void scopy(const unsigned int N, const __fp16 *X, const int incX, __fp16 *Y,
           const int incY) {
  scopy_FP16(N, X, incX, Y, incY);

} // namespace nntrainer

__fp16 snrm2(const int N, const __fp16 *X, const int incX) {
  return snrm2_FP16(N, X, incX);
}

__fp16 sdot(const unsigned int N, const __fp16 *X, const unsigned int incX,
            const __fp16 *Y, const unsigned int incY) {
  return sdot_FP16(N, X, incX, Y, incY);
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const __fp16 *A,
           const unsigned int lda, const __fp16 *X, const int incX,
           const float beta, __fp16 *Y, const int incY) {
  sgemv_FP16(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void sgemv_loop_fp16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                     const unsigned int M, const unsigned int N,
                     const float alpha, const __fp16 *A, const unsigned int lda,
                     const __fp16 *X, const int incX, const float beta,
                     __fp16 *Y, const int incY) {
  std::cout << "sgemv_loop_fp16" << std::endl;
  sgemv_FP16_loop(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

unsigned int isamax(const unsigned int N, const __fp16 *X, const int incX) {
  /// @todo isamax_FP16 for BLAS_NUM_THREADS
  return isamax_FP16(N, X, incX);
}

#endif

#ifndef USE_BLAS
static void saxpy_raw(const unsigned int N, const float alpha, const float *X,
                      const int incX, float *Y, const int incY) {
  if (incX < 0 or incY < 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

static void sgemv_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const int incX, const float beta,
                      float *Y, const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  if (TransA == CblasTrans) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

static float sdot_raw(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

static void scopy_raw(const unsigned int N, const float *X, const int incX,
                      float *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

static void sscal_raw(const unsigned int N, const float alpha, float *X,
                      const int incX) {
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = alpha * X[i * incx];
}

static float snrm2_raw(const unsigned int N, const float *X, const int incX) {
  unsigned int incx = abs(incX);
  float sum = 0.0f;
  float tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

static void sgemm_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      CBLAS_TRANSPOSE TransB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *B, const unsigned int ldb, const float beta,
                      float *C, const unsigned int ldc) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
    }
  }
}

static unsigned int isamax_raw(const unsigned int N, const float *X,
                               const int incX) {

  unsigned int max_idx = 0;
  float max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    float cur_val = abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

#endif

void sscal(const unsigned int N, const float alpha, void *X, const int incX,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  if (d_type == ml::train::TensorDim::DataType::FP32)
    cblas_sscal(N, alpha, (float *)X, incX);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    sscal_raw(N, alpha, (float *)X, incX);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sscal(N, alpha, (__fp16 *)X, incX);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
#endif
}

void sscal(const unsigned int N, const float alpha, float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sscal(N, alpha, X, incX);
#else
  sscal_raw(N, alpha, X, incX);
#endif
}

void saxpy(const unsigned int N, const float alpha, const void *X,
           const int incX, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, static_cast<const float *>(X), incX,
              static_cast<float *>(Y), incY);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    saxpy_raw(N, alpha, static_cast<const float *>(X), incX,
              static_cast<float *>(Y), incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    saxpy_FP16(N, alpha, static_cast<const __fp16 *>(X), incX,
               static_cast<__fp16 *>(Y), incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
#endif
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  saxpy_raw(N, alpha, X, incX, Y, incY);
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const void *A, const unsigned int lda,
           const void *B, const unsigned int ldb, const float beta, void *C,
           const unsigned int ldc, ml::train::TensorDim::DataType d_type) {
#ifdef USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = M * K * sizeof(float);
  unsigned int size_B = K * N * sizeof(float);
  unsigned int size_C = M * N * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
              d_C, N);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#elif defined USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha,
              static_cast<const float *>(A), lda, static_cast<const float *>(B),
              ldb, beta, static_cast<float *>(C), ldc);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    sgemm_raw(order, TransA, TransB, M, N, K, alpha,
              static_cast<const float *>(A), lda, static_cast<const float *>(B),
              ldb, beta, static_cast<float *>(C), ldc);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sgemm_FP16(order, TransA, TransB, M, N, K, alpha,
               static_cast<const __fp16 *>(A), lda,
               static_cast<const __fp16 *>(B), ldb, beta,
               static_cast<__fp16 *>(C), ldc);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {

#ifdef USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = M * K * sizeof(float);
  unsigned int size_B = K * N * sizeof(float);
  unsigned int size_C = M * N * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
              d_C, N);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#elif defined USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#else
  sgemm_raw(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
            ldc);
#endif
}

void scopy(const unsigned int N, const void *X, const int incX, void *Y,
           const int incY, ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    cblas_scopy(N, (float *)X, incX, (float *)Y, incY);
  }
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    scopy_raw(N, (float *)X, incX, (float *)Y, incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    scopy_FP16(N, (__fp16 *)X, incX, (__fp16 *)Y, incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
#endif
} // namespace nntrainer

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_scopy(N, X, incX, Y, incY);
#else
  scopy_raw(N, X, incX, Y, incY);
#endif
} // namespace nntrainer

float snrm2(const int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_snrm2(N, X, incX);
#else
  return snrm2_raw(N, X, incX);
#endif
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sdot(N, X, incX, Y, incY);
#else
  return sdot_raw(N, X, incX, Y, incY);
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const void *A,
           const unsigned int lda, const void *X, const int incX,
           const float beta, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sgemv(order, TransA, M, N, alpha, static_cast<const float *>(A),
                     lda, static_cast<const float *>(X), incX, beta,
                     static_cast<float *>(Y), incY);
#else
  if (d_type == ml::train::TensorDim::DataType::FP32) {
    return sgemv_raw(order, TransA, M, N, alpha, static_cast<const float *>(A),
                     lda, static_cast<const float *>(X), incX, beta,
                     static_cast<float *>(Y), incY);
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    return sgemv_FP16(order, TransA, M, N, alpha,
                      static_cast<const __fp16 *>(A), lda,
                      static_cast<const __fp16 *>(X), incX, beta,
                      static_cast<__fp16 *>(Y), incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                     incY);
#else
  return sgemv_raw(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
}

unsigned int isamax(const unsigned int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_isamax(N, X, incX);
#else
  return isamax_raw(N, X, incX);
#endif
}

/// TODO : DELETE ///
void sgemv_neon(const float *A, const float *X, float *Y, uint32_t rows,
                uint32_t cols, float alpha, float beta) {
  const float *__restrict x;

  for (unsigned int i = 0; i < rows; ++i) {
    Y[i] = Y[i] * beta;
  }

  float32x4_t v_alpha = vmovq_n_f32(alpha);

  if (cols % 16 == 0) {
    for (unsigned i = 0; i < cols; i += 16) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);
      float32x4_t x8_11 = vld1q_f32(&X[i + 8]);
      float32x4_t x12_15 = vld1q_f32(&X[i + 12]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
        x8_11 = vmulq_f32(x8_11, v_alpha);
        x12_15 = vmulq_f32(x12_15, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);
        y0 = vmlaq_f32(y0, wvec8_11, x8_11);
        y0 = vmlaq_f32(y0, wvec12_15, x12_15);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }

  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  } else if (cols % 4 == 0) {
    for (unsigned i = 0; i < cols; i += 4) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  }
}

void sgemv_transpose_neon(const float *A, const float *X, float *Y,
                          uint32_t rows, uint32_t cols, float alpha,
                          float beta) {
  const float *__restrict x;

  const float32x4_t v_beta = vdupq_n_f32(beta);
  const float32x4_t v_alpha = vdupq_n_f32(alpha);

  if (cols % 16 == 0) {
    bool initialized[cols / 16];
    unsigned int step;
    for (unsigned int i = 0; i < cols / 16; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 16) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);
        float32x4_t y8_11 = vld1q_f32(&y[8]);
        float32x4_t y12_15 = vld1q_f32(&y[12]);
        step = j / 16;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          y8_11 = vmulq_f32(y8_11, v_beta);
          y12_15 = vmulq_f32(y12_15, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        y8_11 = vmlaq_f32(y8_11, wvec8_11, x);
        y12_15 = vmlaq_f32(y12_15, wvec12_15, x);

        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
        vst1q_f32(&y[8], y8_11);
        vst1q_f32(&y[12], y12_15);
      }
    }
    return;
  } else if (cols % 8 == 0) {
    bool initialized[cols / 8];
    unsigned int step;
    for (unsigned int i = 0; i < cols / 8; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 8) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);

        step = j / 8;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
      }
    }
    return;
  } else if (cols % 4 == 0) {
    bool initialized[cols / 4];
    unsigned int step;
    for (unsigned int i = 0; i < cols / 4; ++i) {
      initialized[i] = false;
    }
    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 4) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        step = j / 4;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        vst1q_f32(&y[0], y0_3);
      }
    }
  }
  return;
}

void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
  const __fp16 *__restrict x;

  for (unsigned int i = 0; i < rows; ++i) {
    Y[i] = Y[i] * beta;
  }

  float16x8_t v_alpha = vmovq_n_f16(alpha);

  if (cols % 32 == 0) {
    for (unsigned i = 0; i < cols; i += 32) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[i + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[i + 24]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
        x16_23 = vmulq_f16(x16_23, v_alpha);
        x24_31 = vmulq_f16(x24_31, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;

      const __fp16 *__restrict w;

      float16x8_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        __fp16 r[8];
        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);
        y0 = vfmaq_f16(y0, wvec16_23, x16_23);
        y0 = vfmaq_f16(y0, wvec24_31, x24_31);

        vst1q_f16(r, y0);
        for (unsigned int k = 0; k < 8; ++k) {
          Y[j] += r[k];
        }
      }
    }

  } else if (cols % 16 == 0) {

    for (unsigned i = 0; i < cols; i += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15;

      const __fp16 *__restrict w;

      float16x8_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        __fp16 r[8];
        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);

        vst1q_f16(r, y0);
        for (unsigned int k = 0; k < 8; ++k) {
          Y[j] += r[k];
        }
      }
    }
  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
      }

      float16x8_t wvec0_7;

      const __fp16 *__restrict w;

      float16x8_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        __fp16 r[8];
        wvec0_7 = vld1q_f16(&w[0]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);

        vst1q_f16(r, y0);
        for (unsigned int k = 0; k < 8; ++k) {
          Y[j] += r[k];
        }
      }
    }
  }
}

void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta) {
  const __fp16 *__restrict x;

  const float16x8_t v_beta = vmovq_n_f16(beta);
  const float16x8_t v_alpha = vmovq_n_f16(alpha);

  if (cols % 32 == 0) {
    bool initialized[cols / 32];
    unsigned int step;
    for (unsigned int i = 0; i < cols / 32; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float16x8_t x = vld1q_dup_f16(&X[i]);
      x = vmulq_f16(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 32) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);
        float16x8_t y16_23 = vld1q_f16(&y[16]);
        float16x8_t y24_31 = vld1q_f16(&y[24]);

        step = j / 32;
        if (!initialized[step]) {
          y0_7 = vmulq_f16(y0_7, v_beta);
          y8_15 = vmulq_f16(y8_15, v_beta);
          y16_23 = vmulq_f16(y16_23, v_beta);
          y24_31 = vmulq_f16(y24_31, v_beta);
          initialized[step] = true;
        }

        float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0_7 = vfmaq_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_f16(y8_15, wvec8_15, x);
        y16_23 = vfmaq_f16(y16_23, wvec16_23, x);
        y24_31 = vfmaq_f16(y24_31, wvec24_31, x);

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
        vst1q_f16(&y[16], y16_23);
        vst1q_f16(&y[24], y24_31);
      }
    }
    return;
  } else if (cols % 16 == 0) {
    bool initialized[cols / 16];
    unsigned int step;
    for (unsigned int i = 0; i < cols / 16; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float16x8_t x = vld1q_dup_f16(&X[i]);
      x = vmulq_f16(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 16) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);

        step = j / 16;
        if (!initialized[step]) {
          y0_7 = vmulq_f16(y0_7, v_beta);
          y8_15 = vmulq_f16(y8_15, v_beta);
          initialized[step] = true;
        }

        float16x8_t wvec0_7, wvec8_15;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0_7 = vfmaq_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_f16(y8_15, wvec8_15, x);

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
      }
    }
    return;
  } else if (cols % 8 == 0) {
    bool initialized[cols / 8];

    unsigned int step;
    for (unsigned int i = 0; i < cols / 8; ++i) {
      initialized[i] = false;
    }

    __fp16 temp[8];
    for (unsigned int i = 0; i < rows; ++i) {
      float16x8_t x = vld1q_dup_f16(&X[i]);
      x = vmulq_f16(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 8) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);

        step = j / 8;
        if (!initialized[step]) {
          y0_7 = vmulq_f16(y0_7, v_beta);
          initialized[step] = true;
        }

        float16x8_t wvec0_7;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);

        y0_7 = vfmaq_f16(y0_7, wvec0_7, x);

        vst1q_f16(&y[0], y0_7);
      }
    }
    return;
  }
}

} // namespace nntrainer
