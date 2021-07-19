#pragma once
#include <torch/extension.h>

torch::Tensor local_surface(torch::Tensor input,
                            torch::Tensor lengths,
                            double delta_t,
                            int r,
                            double tau);


torch::Tensor local_surface_wrapper(torch::Tensor input,
                                    torch::Tensor lengths,
                                    double delta_t,
                                    int r,
                                    double tau);