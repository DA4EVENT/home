#pragma once
#include <torch/extension.h>

torch::Tensor n_events_cell(torch::Tensor cell_ids,
                            torch::Tensor lengths,
                            int num_cells);

torch::Tensor n_events_cell_wrapper(torch::Tensor cell_ids,
                                    torch::Tensor lengths,
                                    int num_cells);