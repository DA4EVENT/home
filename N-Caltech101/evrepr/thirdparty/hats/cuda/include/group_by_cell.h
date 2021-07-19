#pragma once
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
group_by_cell(torch::Tensor input,
              torch::Tensor cell_ids,
              torch::Tensor lengths,
              int h,
              int w);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
group_by_cell_wrapper(torch::Tensor input,
                      torch::Tensor cell_ids,
                      torch::Tensor lengths,
                      int h,
                      int w);