// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;
  hidden_step_dim.height(to - from);

  Tensor hidden_step = hidden_.getSharedDataTensor(
    hidden_step_dim, from * hidden_dim.width(), true);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    TensorDim input_dim = input_.getDim();

    TensorDim input_step_dim = input_dim;
    input_step_dim.height(to - from);

    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, from * input_dim.width(), true);
    if (!idx) {
      hidden_step.copy(input_step);
    } else {
      hidden_step.add_i(input_step);
    }
  }
  //std::cout <<"addition"<< std::endl;
  //hidden_step.print(std::cout);
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
