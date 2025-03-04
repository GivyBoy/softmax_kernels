#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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

// c++ interface functions that will be called from Python
torch::Tensor naive_softmax_cuda(torch::Tensor input)
{
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
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

    const int threads = 256;
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

    const int threads = 256;
    const int blocks = rows; // One block per row as in original code

    online_softmax_shared_mem_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}