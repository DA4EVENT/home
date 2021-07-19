#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void n_events_cell_kernel(const scalar_t* __restrict__ cell_ids,
                                     const scalar_t* __restrict__ lengths,
									 scalar_t* __restrict__ cell_events_count,
									 const int64_t batch_size,
									 const int64_t event_size,
									 const int64_t num_rf) {

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        for (int64_t e = 0; e < event_size & e < lengths[batch_id]; e++){
            int64_t read_index = batch_id * event_size + e;
            // cell_ids[batch_id][e]
            int64_t cell_id = cell_ids[read_index];
            if (cell_id == -1)
                continue;

            int64_t write_index = batch_id * num_rf + cell_id;
            // cell_events_count[batch_id][cell_id]
            cell_events_count[write_index] += 1;
        }
    }
}


torch::Tensor n_events_cell_wrapper(torch::Tensor cell_ids,
                                    torch::Tensor lengths,
                                    int num_rf){

	// cell_ids.shape = [batch_size, n_events]
	const auto batch_size = cell_ids.size(0);
	const auto event_size = cell_ids.size(1);

	// Create a tensor to hold output counts
	auto options = torch::TensorOptions(at::kLong).device(at::kCUDA);
	auto cell_events_count = torch::zeros({batch_size, num_rf}, options);

    // Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_INTEGRAL_TYPES(cell_ids.type(), "n_events_cell_wrapper", ([&] {
		n_events_cell_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			cell_ids.data_ptr<scalar_t>(),
			lengths.data_ptr<scalar_t>(),
			cell_events_count.data_ptr<scalar_t>(),
			batch_size,
			event_size,
			num_rf);
	}));

	return cell_events_count;
}