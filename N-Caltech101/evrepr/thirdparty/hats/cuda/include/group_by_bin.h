#pragma once
#include <torch/extension.h>

torch::Tensor group_by_bin(torch::Tensor events,
                           torch::Tensor bins_count);


torch::Tensor group_by_bin_wrapper(torch::Tensor events,
                                   torch::Tensor bins_count);