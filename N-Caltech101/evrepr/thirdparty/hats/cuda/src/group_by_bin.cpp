#include "group_by_bin.h"
#include "utils.h"


torch::Tensor group_by_bin(torch::Tensor events,
                           torch::Tensor bins_count){

    CHECK_INPUT(events);
	CHECK_INPUT(bins_count);
	CHECK_IS_LONG(bins_count);

	return group_by_bin_wrapper(events, bins_count);
}
