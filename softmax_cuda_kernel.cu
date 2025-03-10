#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Error checking macro
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

__device__ float fast_exp_ptx(float x)
{
    // we want e^x but ptx only has 2^x, so we can use the fact that e^x = 2^(x * log2(e)), where log2(e) is approx 1.44269504
    // therefore, e^x = 2^(x * 1.44269504)
    // we can use the ex2.approx.f32 instruction to calculate 2^x (note that it is a low precision approximation)
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(x * 1.44269504f));
    return result;
}

__device__ float fast_exp(float x)
{
    return exp2f(x * 1.44269504f);
}

__global__ void naive_softmax_kernel(float *__restrict__ input_matrix, float *__restrict__ output_matrix, int ROWS, int COLS)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ROWS)
    {
        float max_val = -INFINITY;
        float denominator = 0.0f;

        // calc max
        for (int i = 0; i < COLS; i++)
        {
            int idx = row * COLS + i;
            max_val = max(max_val, input_matrix[idx]);
        }

        // calc denominator
        for (int i = 0; i < COLS; i++)
        {
            int idx = row * COLS + i;
            denominator += fast_exp_ptx(input_matrix[idx] - max_val);
        }

        // calc softmax
        for (int i = 0; i < COLS; i++)
        {
            int idx = row * COLS + i;
            output_matrix[idx] = fast_exp_ptx(input_matrix[idx] - max_val) / denominator;
        }
    }
}

/*
ingenius way of reducing the complexity to O(2n) instead of O(3n)

we use an "online" way of carrying out the softmax operation. why not use the power of exponents to our advantage?

we need e^{a_i - max{a_1, ... a_n}} / sum{{e^{a_i - max{a_1, ... a_n}}, ... , e^{a_n - max{a_1, ... a_n}}}}, right?

we can calc the max, and the denominator in a single pass, and then use the results to calc the softmax in another pass. this way, we reduce the complexity to O(2n) instead of O(3n)

how? as i foreshadowed, we can use the power of exponents to our advantage. at any given point, we will have a max val. if we come across a larger value, we would need to update the max val, along with all the previous values.
this isn't entirely hard, since e^{a_i - prev_max} * e^{prev_max - new_max} = e^{a_i - new_max}. eureka!

this is still inefficent, but it's better than the naive approach

*/
__global__ void online_softmax_kernel(float *__restrict__ input_matrix, float *__restrict__ output_matrix, int ROWS, int COLS)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ROWS)
    {
        float max_val = -INFINITY;
        float denominator = 0.0f;

        for (int i = 0; i < COLS; i++)
        {
            int idx = row * COLS + i;
            float cur_max = max_val;

            // calc max
            max_val = max(max_val, input_matrix[idx]);
            float scaling_factor = fast_exp_ptx(cur_max - max_val);

            denominator *= scaling_factor;
            denominator += fast_exp_ptx(input_matrix[idx] - max_val);
        }

        for (int i = 0; i < COLS; i++)
        {
            int idx = row * COLS + i;
            output_matrix[idx] = fast_exp_ptx(input_matrix[idx] - max_val) / denominator;
        }
    }
}

/*

shared memory can be used to reduce the number of reads from global memory, since it is faster (it is the second faster memory after registers). in this case, each block processes a row and each thread processes a chunk/tile of that row

*/
__global__ void online_softmax_shared_mem_kernel(float *__restrict__ input_matrix, float *__restrict__ output_matrix, int ROWS, int COLS)
{
    // shared memory for reductions
    __shared__ float s_mem[1024];

    // each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition
    if (row >= ROWS)
        return;

    // pointers to current row
    float *input_row = input_matrix + row * COLS;
    float *output_row = output_matrix + row * COLS;

    float max_val = -INFINITY;
    float denominator = 0.0f;

    // each thread finds local max and denominator for its elements
    for (int i = tid; i < COLS; i += blockDim.x)
    {
        float x = input_row[i];
        float cur_max = max_val;

        max_val = max(max_val, x);
        float scaling_factor = fast_exp_ptx(cur_max - max_val);

        denominator *= scaling_factor;
        denominator += fast_exp_ptx(x - max_val);
    }
    __syncthreads();

    // find global maximum across all threads
    s_mem[tid] = max_val;
    __syncthreads();

    // tree-based reduction for maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            s_mem[tid] = max(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }

    // get the global max for this row
    float global_max_val = s_mem[0];
    __syncthreads();

    // adjust local denominators based on global max
    s_mem[tid] = denominator * fast_exp_ptx(max_val - global_max_val);
    __syncthreads();

    // tree-based reduction for denominator
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    // get the global denominator
    float global_denominator = s_mem[0];
    __syncthreads();

    // compute final softmax values
    for (int i = tid; i < COLS; i += blockDim.x)
    {
        output_row[i] = fast_exp_ptx(input_row[i] - global_max_val) / global_denominator;
    }
}

__global__ void online_softmax_shared_mem_vectorized_kernel(const float *__restrict__ input_matrix,
                                                            float *__restrict__ output_matrix,
                                                            int ROWS, int COLS)
{
    // each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= ROWS)
        return;

    // pointers to current row
    const float *row_in = input_matrix + row * COLS;
    float *row_out = output_matrix + row * COLS;

    // calculate how many float4 chunks we need - assume COLS % 4 == 0
    int vecCols = COLS / 4;
    int remainder = COLS % 4;

    extern __shared__ float sdata[];

    // compute max
    float local_max = -INFINITY;
    for (int i = tid; i < vecCols; i += blockDim.x)
    {
        // Load 4 elements at once
        float4 data = reinterpret_cast<const float4 *>(row_in)[i];
        local_max = fmaxf(local_max, data.x);
        local_max = fmaxf(local_max, data.y);
        local_max = fmaxf(local_max, data.z);
        local_max = fmaxf(local_max, data.w);
    }

    // store each thread's local max into shared memory.
    sdata[tid] = local_max;
    __syncthreads();

    // reduction in shared memory to get global max.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float global_max = sdata[0];
    __syncthreads();

    // compute the sum of exp(x - global_max)
    float local_sum = 0.0f;
    for (int i = tid; i < vecCols; i += blockDim.x)
    {
        float4 data = reinterpret_cast<const float4 *>(row_in)[i];
        local_sum += expf(data.x - global_max);
        local_sum += expf(data.y - global_max);
        local_sum += expf(data.z - global_max);
        local_sum += expf(data.w - global_max);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // reduction in shared memory to sum up denominators.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float global_sum = sdata[0];
    __syncthreads();

    // compute the final softmax output in vectorized form
    for (int i = tid; i < vecCols; i += blockDim.x)
    {
        float4 data = reinterpret_cast<const float4 *>(row_in)[i];
        float4 result;
        result.x = expf(data.x - global_max) / global_sum;
        result.y = expf(data.y - global_max) / global_sum;
        result.z = expf(data.z - global_max) / global_sum;
        result.w = expf(data.w - global_max) / global_sum;
        reinterpret_cast<float4 *>(row_out)[i] = result;
    }
}

__global__ void online_softmax_vectorized_kernel(float *__restrict__ input_matrix,
                                                 float *__restrict__ output_matrix,
                                                 int ROWS, int COLS)
{
    // shared memory for reductions
    __shared__ float s_mem[1024];

    // each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= ROWS)
        return;

    float *input_row = input_matrix + row * COLS;
    float *output_row = output_matrix + row * COLS;

    float max_val = -INFINITY;
    float denominator = 0.0f;

    // each thread processes 4 elements at once where possible
    for (int i = tid * 4; i < COLS; i += blockDim.x * 4)
    {
        // load up to 4 elements at once
        float4 data;

        // handle boundary conditions
        if (i < COLS)
            data.x = input_row[i];
        else
            data.x = -INFINITY;

        if (i + 1 < COLS)
            data.y = input_row[i + 1];
        else
            data.y = -INFINITY;

        if (i + 2 < COLS)
            data.z = input_row[i + 2];
        else
            data.z = -INFINITY;

        if (i + 3 < COLS)
            data.w = input_row[i + 3];
        else
            data.w = -INFINITY;

        float cur_max;

        if (i < COLS)
        {
            cur_max = max_val;
            max_val = max(max_val, data.x);
            float scaling_factor = fast_exp_ptx(cur_max - max_val);
            denominator *= scaling_factor;
            denominator += fast_exp_ptx(data.x - max_val);
        }

        if (i + 1 < COLS)
        {
            cur_max = max_val;
            max_val = max(max_val, data.y);
            float scaling_factor = fast_exp_ptx(cur_max - max_val);
            denominator *= scaling_factor;
            denominator += fast_exp_ptx(data.y - max_val);
        }

        if (i + 2 < COLS)
        {
            cur_max = max_val;
            max_val = max(max_val, data.z);
            float scaling_factor = fast_exp_ptx(cur_max - max_val);
            denominator *= scaling_factor;
            denominator += fast_exp_ptx(data.z - max_val);
        }

        if (i + 3 < COLS)
        {
            cur_max = max_val;
            max_val = max(max_val, data.w);
            float scaling_factor = fast_exp_ptx(cur_max - max_val);
            denominator *= scaling_factor;
            denominator += fast_exp_ptx(data.w - max_val);
        }
    }
    __syncthreads();

    // find global maximum across all threads
    s_mem[tid] = max_val;
    __syncthreads();

    // tree-based reduction for maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            s_mem[tid] = max(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }

    // global max for this row
    float global_max_val = s_mem[0];
    __syncthreads();

    // adjust local denominators based on global max
    s_mem[tid] = denominator * fast_exp_ptx(max_val - global_max_val);
    __syncthreads();

    // tree-based reduction for denominator
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    // global denominator
    float global_denominator = s_mem[0];
    __syncthreads();

    // compute final softmax values - process 4 elements at once where possible
    for (int i = tid * 4; i < COLS; i += blockDim.x * 4)
    {
        // Process with boundary handling
        if (i < COLS)
            output_row[i] = fast_exp_ptx(input_row[i] - global_max_val) / global_denominator;

        if (i + 1 < COLS)
            output_row[i + 1] = fast_exp_ptx(input_row[i + 1] - global_max_val) / global_denominator;

        if (i + 2 < COLS)
            output_row[i + 2] = fast_exp_ptx(input_row[i + 2] - global_max_val) / global_denominator;

        if (i + 3 < COLS)
            output_row[i + 3] = fast_exp_ptx(input_row[i + 3] - global_max_val) / global_denominator;
    }
}

__global__ void online_softmax_warp_optimized_kernel(float *__restrict__ input_matrix,
                                                     float *__restrict__ output_matrix,
                                                     int ROWS, int COLS)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // warp params
    const int WARP_SIZE = 32;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    if (row >= ROWS)
        return;

    float *input_row = input_matrix + row * COLS;
    float *output_row = output_matrix + row * COLS;

    // shared memory for cross-warp communication
    __shared__ float s_warp_max[32]; // Assumes at most 32 warps per block
    __shared__ float s_warp_denom[32];

    float max_val = -INFINITY;
    float denominator = 0.0f;

    // each thread finds local max and denominator for its elements
    for (int i = tid; i < COLS; i += blockDim.x)
    {
        float x = input_row[i];
        float cur_max = max_val;

        max_val = max(max_val, x);
        float scaling_factor = fast_exp_ptx(cur_max - max_val);

        denominator *= scaling_factor;
        denominator += fast_exp_ptx(x - max_val);
    }

    // warp-level reduction for max using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = max(max_val, other_max);
    }

    // first thread in each warp writes its result to shared memory
    if (laneId == 0)
    {
        s_warp_max[warpId] = max_val;
    }
    __syncthreads();

    // first warp finds the global max across all warps
    if (warpId == 0)
    {
        if (laneId < blockDim.x / WARP_SIZE)
        {
            max_val = s_warp_max[laneId];
        }
        else
        {
            max_val = -INFINITY;
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
            max_val = max(max_val, other_max);
        }

        if (laneId == 0)
        {
            s_warp_max[0] = max_val;
        }
    }
    __syncthreads();

    // global max for the row
    float global_max_val = s_warp_max[0];

    // adjust local denominator based on global max
    denominator *= fast_exp_ptx(max_val - global_max_val);

    // warp-level reduction for denominator using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        denominator += __shfl_down_sync(0xffffffff, denominator, offset);
    }

    // first thread in each warp writes its result to shared memory
    if (laneId == 0)
    {
        s_warp_denom[warpId] = denominator;
    }
    __syncthreads();

    // first warp finds the global denominator across all warps
    if (warpId == 0)
    {
        if (laneId < blockDim.x / WARP_SIZE)
        {
            denominator = s_warp_denom[laneId];
        }
        else
        {
            denominator = 0.0f;
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            denominator += __shfl_down_sync(0xffffffff, denominator, offset);
        }

        if (laneId == 0)
        {
            s_warp_denom[0] = denominator;
        }
    }
    __syncthreads();

    // global denominator for the row
    float global_denominator = s_warp_denom[0];

    // compute final softmax values
    for (int i = tid; i < COLS; i += blockDim.x)
    {
        output_row[i] = fast_exp_ptx(input_row[i] - global_max_val) / global_denominator;
    }
}

__global__ void online_softmax_cooperative_kernel(float *__restrict__ input_matrix,
                                                  float *__restrict__ output_matrix,
                                                  int ROWS, int COLS)
{
    // Get cooperative groups
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= ROWS)
        return;

    float *input_row = input_matrix + row * COLS;
    float *output_row = output_matrix + row * COLS;

    // shared memory for cross-warp communication
    __shared__ float s_warp_max[32]; // Assumes at most 32 warps per block
    __shared__ float s_warp_denom[32];

    // warp id
    int warpId = tid / 32;
    int laneId = tid % 32;

    float max_val = -INFINITY;
    float denominator = 0.0f;

    // each thread finds local max and denominator for its elements
    for (int i = tid; i < COLS; i += block.size())
    {
        float x = input_row[i];
        float cur_max = max_val;

        max_val = max(max_val, x);
        float scaling_factor = fast_exp_ptx(cur_max - max_val);

        denominator *= scaling_factor;
        denominator += fast_exp_ptx(x - max_val);
    }

    // warp-level reduction for max using shuffle
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_max = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = max(max_val, other_max);
    }

    // first thread in each warp writes its result to shared memory
    if (laneId == 0)
    {
        s_warp_max[warpId] = max_val;
    }
    block.sync();

    // first warp reduces across all warps
    if (warpId == 0)
    {
        float warp_group_max = -INFINITY;

        if (laneId < (block.size() + 31) / 32) // # of warps
        {
            warp_group_max = s_warp_max[laneId];
        }

        // reduce max within the first warp manually
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float other_max = __shfl_down_sync(0xffffffff, warp_group_max, offset);
            warp_group_max = max(warp_group_max, other_max);
        }

        // first thread writes the result for all to see
        if (laneId == 0)
        {
            s_warp_max[0] = warp_group_max;
        }
    }
    block.sync();

    // global max for the row
    float global_max_val = s_warp_max[0];

    // adjust local denominator based on global max
    denominator *= fast_exp_ptx(max_val - global_max_val);

    // manual warp-level reduction for denominator
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_denom = __shfl_down_sync(0xffffffff, denominator, offset);
        denominator += other_denom;
    }

    // first thread in each warp writes its result to shared memory
    if (laneId == 0)
    {
        s_warp_denom[warpId] = denominator;
    }
    block.sync();

    // first warp reduces across all warps
    if (warpId == 0)
    {
        float warp_group_denom = 0.0f;

        if (laneId < (block.size() + 31) / 32) // Number of warps
        {
            warp_group_denom = s_warp_denom[laneId];
        }

        for (int offset = 16; offset > 0; offset /= 2)
        {
            float other_denom = __shfl_down_sync(0xffffffff, warp_group_denom, offset);
            warp_group_denom += other_denom;
        }

        if (laneId == 0)
        {
            s_warp_denom[0] = warp_group_denom;
        }
    }
    block.sync();

    // global denominator for the row
    float global_denominator = s_warp_denom[0];

    // compute final softmax values
    for (int i = tid; i < COLS; i += block.size())
    {
        output_row[i] = fast_exp_ptx(input_row[i] - global_max_val) / global_denominator;
    }
}

// c++ interface functions that will be called from Python
torch::Tensor naive_softmax_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = (rows + threads - 1) / threads;

    naive_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = (rows + threads - 1) / threads;

    online_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_shared_mem_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = rows; // one block per row

    online_softmax_shared_mem_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_shared_mem_vectorized_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = rows; // one block per row

    // calculate required shared memory size (in bytes)
    // need threads * sizeof(float) bytes for reduction
    const int smem_size = threads * sizeof(float);

    online_softmax_shared_mem_vectorized_kernel<<<blocks, threads, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_vectorized_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 512;
    const int blocks = rows; // one block per row

    online_softmax_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_warp_optimized_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 512;
    const int blocks = rows; // one block per row

    online_softmax_warp_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor online_softmax_cooperative_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = rows; // one block per row

    online_softmax_cooperative_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}