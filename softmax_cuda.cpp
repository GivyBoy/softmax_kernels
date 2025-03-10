#include <torch/extension.h>

// forward declarations of CUDA functions
torch::Tensor naive_softmax_cuda(torch::Tensor input);
torch::Tensor online_softmax_cuda(torch::Tensor input);
torch::Tensor online_softmax_shared_mem_cuda(torch::Tensor input);
torch::Tensor online_softmax_shared_mem_vectorized_cuda(torch::Tensor input);
torch::Tensor online_softmax_vectorized_cuda(torch::Tensor input);
torch::Tensor online_softmax_warp_optimized_cuda(torch::Tensor input);
torch::Tensor online_softmax_cooperative_cuda(torch::Tensor input);

// wrapper functions that check input types and dimensions
torch::Tensor naive_softmax(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return naive_softmax_cuda(input);
}

torch::Tensor online_softmax(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_cuda(input);
}

torch::Tensor online_softmax_shared_mem(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_shared_mem_cuda(input);
}

torch::Tensor online_softmax_shared_mem_vectorized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_shared_mem_vectorized_cuda(input);
}

torch::Tensor online_softmax_vectorized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_vectorized_cuda(input);
}

torch::Tensor online_softmax_warp_optimized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_warp_optimized_cuda(input);
}

torch::Tensor online_softmax_cooperative(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (rows x cols)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input tensor must be float32");
    
    return online_softmax_cooperative_cuda(input);
}

// Define Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_softmax", &naive_softmax, "Naive Softmax implementation (CUDA)");
    m.def("online_softmax", &online_softmax, "Online Softmax implementation (CUDA)");
    m.def("online_softmax_shared_mem", &online_softmax_shared_mem, "Online Softmax with shared memory (CUDA)");
    m.def("online_softmax_shared_mem_vectorized", &online_softmax_shared_mem_vectorized, "Online Softmax with shared memory and vectorization (CUDA)");
    m.def("online_softmax_vectorized", &online_softmax_vectorized, "Online Softmax with manual vectorization (CUDA)");
    m.def("online_softmax_warp_optimized", &online_softmax_warp_optimized, "Online Softmax with warp-level optimizations (CUDA)");
    m.def("online_softmax_cooperative", &online_softmax_cooperative, "Online Softmax with cooperative groups (CUDA)");
}