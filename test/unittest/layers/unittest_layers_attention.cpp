// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_attention.cpp
 * @date 1 October 2021
 * @brief Attention Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <attention_layer.h>
#include <layers_common_tests.h>

auto semantic_attention = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>,
  nntrainer::AttentionLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(Attention, LayerSemantics,
                     ::testing::Values(semantic_attention));

auto attention_shared_kv = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "1:1:5:7,1:1:3:7",
  "attention_shared_kv.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto attention_shared_kv_batched = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "2:1:5:7,2:1:3:7",
  "attention_shared_kv_batched.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto attention_batched = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {},
  "2:1:5:7,2:1:3:7,2:1:3:7", "attention_batched.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto attention_shared_kv_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "1:7:1:5,1:7:1:3",
  "attention_shared_kv.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

auto attention_shared_kv_batched_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "2:7:1:5,2:7:1:3",
  "attention_shared_kv_batched.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

auto attention_batched_nhwc = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {},
  "2:7:1:5,2:7:1:3,2:7:1:3", "attention_batched.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nhwc", "fp32", "fp32");

GTEST_PARAMETER_TEST(
  Attention, LayerGoldenTest,
  ::testing::Values(attention_shared_kv, attention_shared_kv_batched,
                    attention_batched, attention_shared_kv_nhwc,
                    attention_shared_kv_batched_nhwc, attention_batched_nhwc));

#ifdef ENABLE_FP16
auto attention_shared_kv_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "1:1:5:7,1:1:3:7",
  "attention_shared_kv_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto attention_shared_kv_batched_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "2:1:5:7,2:1:3:7",
  "attention_shared_kv_batched_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto attention_batched_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {},
  "2:1:5:7,2:1:3:7,2:1:3:7", "attention_batched_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(Attention16, LayerGoldenTest,
                     ::testing::Values(attention_shared_kv_w16a16));
#endif
