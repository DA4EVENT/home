#include "n_events_cell.h"
#include "utils.h"


torch::Tensor n_events_cell(torch::Tensor cell_ids,
                            torch::Tensor lengths,
                            int num_cells){
	CHECK_INPUT(cell_ids);
	CHECK_IS_LONG(cell_ids);
	CHECK_INPUT(lengths);
	CHECK_IS_LONG(lengths);

	return n_events_cell_wrapper(cell_ids, lengths, num_cells);
}
