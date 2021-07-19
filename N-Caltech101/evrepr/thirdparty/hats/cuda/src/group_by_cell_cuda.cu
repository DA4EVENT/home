#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <n_events_cell.h>


template <typename scalar_t>
__global__ void group_by_cell_kernel(const scalar_t* __restrict__ input,
                                     const int64_t* __restrict__ cell_ids,
			                         const int64_t* __restrict__ lengths,
			                         const int64_t* __restrict__ cell_offsets,
			                         scalar_t* __restrict__ groups,
			                         int64_t* __restrict__ gr_len,
                                     int64_t* __restrict__ gr_batch_id,
                                     int64_t* __restrict__ gr_h,
                                     int64_t* __restrict__ gr_w,
                                     const int64_t out_w,
			                         const int64_t batch_size,
			                         const int64_t event_size,
			                         const int64_t feature_size,
			                         const int64_t num_rf,
			                         const int64_t new_batch_size){

    const int64_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id < batch_size){
        for (int64_t e = 0; e < event_size & e < lengths[batch_id]; e++){
            // cell_idx[batch_id][e]
            const int64_t cell_id = cell_ids[batch_id * event_size + e];
            if (cell_id == -1)
                continue;
            // Retrieve the position of the receptive field within the output tensor
            // cell_offsets[batch_id][cell_id]
            const int64_t cell_pos = cell_offsets[batch_id * num_rf + cell_id] - 1;
            // Retrieve the number of events already placed inside the target rf
            // const int64_t event_pos = cell_event_counts[cell_id];
            const int64_t event_pos = gr_len[cell_pos];

            // groups[event_pos][cell_pos][:] = input[batch_id][e][:]
            const int64_t write_offset = event_pos * (new_batch_size * feature_size) \
                                         + cell_pos * feature_size;
            const int64_t read_offset = batch_id * (event_size * feature_size) \
                                        + e * feature_size;
            for (int64_t f = 0; f < feature_size; f++)
                groups[write_offset + f] = input[read_offset + f];

            // Increment the number of events in the current receptive field
            gr_len[cell_pos] += 1;

            // We write receptive field information just one time (it is the same for
            // all the events in the same receptive field), ie, the first time we process
            // an event inside the receptive field
            if (event_pos == 0){
                gr_batch_id[cell_pos] = batch_id;
                gr_h[cell_pos] = (int)(cell_id / out_w);
                gr_w[cell_pos] = (int)(cell_id % out_w);
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
group_by_cell_wrapper(torch::Tensor input, 
                      torch::Tensor cell_ids,
                      torch::Tensor lengths, 
                      int h, 
                      int w){

    const auto batch_size = input.size(0);
    const auto event_size = input.size(1);
    const auto feature_size = input.size(2);

    // Compute the number of events in each receptive field of every sample
    const auto cell_counts = n_events_cell_wrapper(cell_ids, lengths, h*w);
    // Compute the new event length (max events in a receptive field)
    const auto new_event_size = torch::max(cell_counts).cpu().item().to<int64_t>();
    // Compute the position that each non empty receptive field will have in 
    // the output tensor
    const auto cell_offsets = torch::cumsum(
        cell_counts.gt(0).view({-1}), /*dim=*/-1).view(cell_counts.sizes());
    // Compute the new flat batch size (tot num of non empty receptive fields)
    const auto new_batch_size = cell_offsets[batch_size-1][h*w-1].cpu().item().to<int64_t>();

    // Create a tensor to hold output groups
	auto groups = torch::zeros({new_event_size, new_batch_size, feature_size}, input.options());
	auto gr_len = torch::zeros({new_batch_size}, input.options().dtype(at::kLong));
	auto gr_batch_id = torch::zeros({new_batch_size}, input.options().dtype(at::kLong));
	auto gr_h = torch::zeros({new_batch_size}, input.options().dtype(at::kLong));
	auto gr_w = torch::zeros({new_batch_size}, input.options().dtype(at::kLong));

	// Allocate a thread for each sample, each one will process all the events
    // in the sample (just the batch loop is parallelized)
	int threadsPerBlock = 32;
	int numBlocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

	AT_DISPATCH_ALL_TYPES(input.type(), "group_by_cell_wrapper", ([&] {
		group_by_cell_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			input.data_ptr<scalar_t>(),
			cell_ids.data_ptr<int64_t>(),
			lengths.data_ptr<int64_t>(),
			cell_offsets.data_ptr<int64_t>(),
			groups.data_ptr<scalar_t>(),
			gr_len.data_ptr<int64_t>(),
			gr_batch_id.data_ptr<int64_t>(),
			gr_h.data_ptr<int64_t>(),
			gr_w.data_ptr<int64_t>(),
			w,
			batch_size,
			event_size,
			feature_size,
			h*w,
			new_batch_size);
	}));

    return std::make_tuple(gr_batch_id, gr_len, gr_h, gr_w, groups);
}
