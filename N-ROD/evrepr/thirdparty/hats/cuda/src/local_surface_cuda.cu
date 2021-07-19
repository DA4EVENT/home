#include <torch/extension.h>
#include <vector>
#include <stdio.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void local_surface_kernel(const scalar_t* __restrict__ input,
                                     const int64_t* __restrict__ lengths,
                                     float* __restrict__ output,
			                         const int64_t event_size,
                                     const int64_t batch_size,
			                         const int64_t feature_size,
                                     const double delta_t,
                                     const int64_t r,
                                     const double tau){

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int rf_size = 2 * r + 1;

    if (i < event_size & b < batch_size){
        int64_t i_idx = i * batch_size * feature_size + b * feature_size;
        int i_x = input[i_idx + 0];
        int i_y = input[i_idx + 1];
        auto i_t = input[i_idx + 2];
        int i_p = input[i_idx + 3];
        // Convert -1 polarity to 0
        i_p = i_p < 0 ? 0 : i_p;

        if (i < lengths[b]){
            // Look into the past memory
            for(int64_t j = i - 1; j >= 0; j--){
                int64_t j_idx = j * batch_size * feature_size + b * feature_size;
                int j_x = input[j_idx + 0];
                int j_y = input[j_idx + 1];
                auto j_t = input[j_idx + 2];
                int j_p = input[j_idx + 3];
                // Convert -1 polarity to 0
                j_p = j_p < 0 ? 0 : j_p;

                // We stop as soon as we exit the event's memory
                if (j_t < i_t - delta_t)
                    break;

                int rf_y = j_y - i_y + r;
                int rf_x = j_x - i_x + r;

                // The jth event is inside the ith event's neighborhood
                if ((j_t < i_t) & (j_p == i_p) &
                    (rf_x >= 0) & (rf_x < rf_size) &
                    (rf_y >= 0) & (rf_y < rf_size)){

                    float value = expf((float)(j_t - i_t) / tau);
                    int64_t out_idx = i * batch_size * 2 * rf_size * rf_size + \
                                      b * 2 * rf_size * rf_size + \
                                      i_p * rf_size * rf_size + \
                                      rf_y * rf_size + \
                                      rf_x;
                    output[out_idx] = output[out_idx] + value;
                }
            }
        }
    }
}


torch::Tensor local_surface_wrapper(torch::Tensor input,
                                    torch::Tensor lengths,
                                    double delta_t,
                                    int r,
                                    double tau){

    const auto event_size = input.size(0);
    const auto batch_size = input.size(1);
    const auto feature_size = input.size(2);

    // Create a tensor to hold the result
	auto output = torch::zeros({event_size, batch_size, 2, 2*r + 1, 2*r + 1},
	                            input.options().dtype(at::kFloat));

    // Split the first dimension over threadsPerBlock.x threads
    // and the second dimension over threadsPerBlock.y threads
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int)((event_size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                   (int)((batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y));

    AT_DISPATCH_ALL_TYPES(input.type(), "local_surface_wrapper", ([&] {
		local_surface_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			input.data_ptr<scalar_t>(),
			lengths.data_ptr<int64_t>(),
			output.data_ptr<float>(),
			event_size,
			batch_size,
			feature_size,
			delta_t,
			r,
			tau);
	}));

    return output;
}