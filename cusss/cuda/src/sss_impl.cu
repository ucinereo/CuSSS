#include "sss_impl.hpp"

#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

__global__ void sss_forward_kernel(const float* x, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        output[idx] = (e * inv) * 0.5f + 0.5f;
    }
}

__global__ void sss_backward_kernel(const float* x, const float* grad_out, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        float grad = inv * inv * 0.5f;
        grad_x[idx] = grad_out[idx] * grad;
    }
}


at::Tensor sss_forward_cuda(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");

    auto out = at::empty_like(x);

    int size = x.numel();
    std::cout << "[sss_forward_cuda] called, numel=" << x.numel() << std::endl;

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // <<<blocks, threads>>>
    sss_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        x.numel()
    );

    return out;
}

at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output) {
    TORCH_CHECK(x.is_cuda() && grad_output.is_cuda(), "CUDA only");

    auto grad_x = at::empty_like(x);

    int size = x.numel();
    std::cout << "[sss_backward_cuda] called, numel=" << x.numel() << std::endl;


    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;


    // <<<blocks, threads>>>
    sss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        x.numel()
    );

    return grad_x;
}




