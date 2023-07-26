// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_nntrainer_tensor.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for tensor.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(nntrainer_Tensor, sum_02_p) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor ans0(
    std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
      {{{{39, 42, 45, 48, 51, 54, 57, 60, 63, 66},
         {69, 72, 75, 78, 81, 84, 87, 90, 93, 96}},
        {{57, 60, 63, 66, 69, 72, 75, 78, 81, 84},
         {87, 90, 93, 96, 99, 102, 105, 108, 111, 114}}}}),
    t_type_nchw_fp16);

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_copy(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-1;
  const float epsilon = 1e-3;
  float mse = 0;
  float sum1 = 0;
  float sum2 = 0;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result0_loop = input_copy.sum_loop(0);
  nntrainer::Tensor result0_fp32 = input_fp32.sum(0);

  float error1 = 0;
  float error2 = 0;
  float error3 = 0;
  float error4 = 0;
  float error5 = 0;

  for (unsigned int i = 0; i < result0.batch(); ++i) {
    for (unsigned int c = 0; c < result0.channel(); ++c) {
      for (unsigned int j = 0; j < result0.height(); ++j) {
        for (unsigned int k = 0; k < result0.width(); ++k) {

          // FP32
          float val1 = result0_fp32.getValue<float>(i, c, j, k);

          // (FP16 casted) FP32
          __fp16 val2 = static_cast<__fp16>(val1);

          // NEON FP16
          __fp16 val3 = result0.getValue<__fp16>(i, c, j, k);

          // LOOP FP16
          __fp16 val4 = result0_loop.getValue<__fp16>(i, c, j, k);

          // FP32 VS CASTED FP32
          if (abs(val1 - val2) > epsilon) {
            std::cout << val1 << " VS " << val2 << std::endl;
            std::cout << "ERROR : " << abs(val1 - val2) << std::endl;
            error1 += abs(val1 - val2);
          }

          // FP32 VS NEON FP16
          if (abs(val1 - val3) > epsilon) {
            std::cout << val1 << " VS " << val3 << std::endl;
            std::cout << "ERROR : " << abs(val1 - val3) << std::endl;
            error2 += abs(val1 - val3);
          }

          // FP32 VS LOOP FP16
          if (abs(val1 - val4) > epsilon) {
            std::cout << val1 << " VS " << val4 << std::endl;
            std::cout << "ERROR : " << abs(val1 - val4) << std::endl;
            error3 += abs(val1 - val4);
          }

          // NEON FP16 VS LOOP FP16
          if (abs(val3 - val4) > epsilon) {
            std::cout << val3 << " VS " << val4 << std::endl;
            std::cout << "ERROR : " << abs(val3 - val4) << std::endl;
            error4 += abs(val3 - val4);
          }

          // NEON FP16 VS CASTED FP16
          if (abs(val2 - val3) > epsilon) {
            std::cout << val2 << " VS " << val3 << std::endl;
            std::cout << "ERROR : " << abs(val2 - val3) << std::endl;
            error5 += abs(val2 - val3);
          }
          // LOOP FP16 VS CASTED FP16
          if (abs(val2 - val4) > epsilon) {
            std::cout << val2 << " VS " << val4 << std::endl;
            std::cout << "ERROR : " << abs(val2 - val4) << std::endl;
          }
        }
      }
    }
  }
  std::cout << "Error 1,2,3,4,5 : " << error1 << " " << error2 << " " << error3
            << " " << error4 << " " << error5 << std::endl;
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
