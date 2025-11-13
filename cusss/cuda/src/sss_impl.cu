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


// ===================================================================
// FORWARD AND BACKWARD IMPLEMENTATIONS

torch::Tensor forward_cuda(torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    auto output = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}

std::vector<torch::Tensor> backward_cuda(torch::Tensor &x, torch::Tensor &grad_outputs) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must be a CUDA tensor!");
    auto grad_x = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_outputs.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    return {grad_x};
}


// ===================================================================
// AUTOGRAD CLASS DEFINITIONS

torch::Tensor SSSAutograd::forward(AutogradContext *ctx, Tensor x) {
    ctx->save_for_backward({x});
    return forward_cuda(x);
}

variable_list SSSAutograd::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    return backward_cuda(x, grad_outputs[0]);
}
