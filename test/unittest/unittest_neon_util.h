// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.h
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is header for blas neon implementation
 *
 */

#include "arm_neon.h"

/**
 * @brief     sgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_neon(uint32_t rows, uint32_t cols, const float alpha, const float *A,
                const unsigned int lda, const float *X, const int incX,
                const float beta, float *Y, const int incY);

/**
 * @brief     transposed sgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_transpose_neon(uint32_t rows, uint32_t cols, const float alpha,
                          const float *A, const unsigned int lda,
                          const float *X, const int incX, const float beta,
                          float *Y, const int incY);

/**
 * @brief     sgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_neon_fp16(uint32_t rows, uint32_t cols, const float alpha,
                     const __fp16 *A, const unsigned int lda, const __fp16 *X,
                     const int incX, const float beta, __fp16 *Y,
                     const int incY);

/**
 * @brief     transposed sgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_transpose_neon_fp16(uint32_t rows, uint32_t cols, const float alpha,
                               const __fp16 *A, const unsigned int lda,
                               const __fp16 *X, const int incX,
                               const float beta, __fp16 *Y, const int incY);