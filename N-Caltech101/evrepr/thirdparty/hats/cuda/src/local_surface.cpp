#include "local_surface.h"
#include "utils.h"


torch::Tensor local_surface(torch::Tensor input,
                            torch::Tensor lengths,
                            double delta_t,
                            int r,
                            double tau){

    CHECK_INPUT(input);
    CHECK_INPUT(lengths);
    CHECK_IS_LONG(lengths);

    return local_surface_wrapper(input, lengths, delta_t, r, tau);
}
