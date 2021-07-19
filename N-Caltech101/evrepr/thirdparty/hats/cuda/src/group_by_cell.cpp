#include "group_by_cell.h"
#include "utils.h"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
group_by_cell(torch::Tensor input,
              torch::Tensor cell_ids,
              torch::Tensor lengths,
              int h,
              int w){

    CHECK_INPUT(input);
    CHECK_INPUT(cell_ids);
    CHECK_IS_LONG(cell_ids);
    CHECK_INPUT(lengths);
    CHECK_IS_LONG(lengths);

    return group_by_cell_wrapper(input, cell_ids, lengths, h, w);
}
