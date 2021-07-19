#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <n_events_cell.h>


template <typename scalar_t>
__global__ void group_by_bin_kernel(const scalar_t* events, 
                                    const int64_t* bins_count,
                                    int64_t batch_size, 
                                    int64_t event_size, 
                                    int64_t feature_size,
                                    int64_t n_intervals, 
                                    int64_t new_event_size,
                                    scalar_t* new_events) {

    const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int interval_id = blockIdx.y * blockDim.y + threadIdx.y;
	const int event_id = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_id < batch_size & interval_id < n_intervals & event_id < new_event_size) {
	    // bins_count[batch_id][interval_id]
        int interval_len = bins_count[batch_id * n_intervals + interval_id];
	    if (event_id < interval_len){
            int offset = 0;
            for (int i = 0; i < interval_id; i++){
                // bins_count[batch_id][i]
                offset += bins_count[batch_id * n_intervals + i];
            }
            if ((event_id + offset) < event_size) {
                auto write_offset = batch_id * n_intervals * (new_event_size * feature_size) + \
                                    interval_id * (new_event_size * feature_size) + \
                                    event_id * feature_size;
                auto read_offset = batch_id * (event_size * feature_size) + offset * feature_size + \
                                   event_id * feature_size;

                for (int f = 0; f < feature_size; f++){
                    // new_events[batch_id + interval_id * n_intervals]
                    new_events[write_offset + f] = events[read_offset + f];
                }
            }
        }
	}
}


torch::Tensor group_by_bin_wrapper(torch::Tensor events,
                                   torch::Tensor bins_count){

    // events.shape = [batch_size, n_events, features]
	const auto batch_size = events.size(0);
	const auto event_size = events.size(1);
	const auto feature_size = events.size(2);
	// bins_count.shape = [batch_size, n_intervals]
	const auto n_intervals = bins_count.size(1);

    const auto new_event_size = torch::max(bins_count).cpu().item().to<int64_t>();
	auto new_events = torch::zeros({batch_size * n_intervals,
                                    new_event_size, feature_size}, events.options());

	dim3 threadsPerBlock(4, 4, 64);
    dim3 numBlocks((int)((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((n_intervals + threadsPerBlock.y - 1) / threadsPerBlock.y),
                   (int)((new_event_size + threadsPerBlock.z - 1) / threadsPerBlock.z));

	AT_DISPATCH_ALL_TYPES(events.type(), "group_by_bin", ([&] {
		group_by_bin_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			events.data<scalar_t>(),
			bins_count.data<int64_t>(),
			batch_size,
			event_size,
			feature_size,
			n_intervals,
			new_event_size,
			new_events.data<scalar_t>());
	}));

	return new_events;
}
