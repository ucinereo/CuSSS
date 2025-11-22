#include "xsss_impl.hpp"

#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

__global__ void xsss_forward_kernel(const float* x, const float* a, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        output[idx] = (e * inv) * a[0] + 0.5f;
    }
}

__global__ void xsss_backward_kernel(const float* x, const float* a, const float* grad_out, float* grad_x, float* grad_a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));

        float local_grad_x = inv * inv * a[0];
        grad_x[idx] = grad_out[idx] * local_grad_x;
        
        float local_grad_a = grad_out[idx] * e * inv;
        atomicAdd(grad_a, local_grad_a);
    }
}

// ===================================================================
// FORWARD AND BACKWARD IMPLEMENTATIONS

torch::Tensor forward_cuda(const torch::Tensor &x, const torch::Tensor &a) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input x tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(a.dtype() == torch::kFloat, "Input a tensor must be a float!");
    TORCH_CHECK(a.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");

    auto output = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    xsss_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), a.data_ptr<float>(), output.data_ptr<float>(), size
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "xsss_forward_kernel launch failed");


    return output;
}

std::vector<torch::Tensor> backward_cuda(const torch::Tensor &x, const torch::Tensor &a, const torch::Tensor &grad_outputs) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(a.dtype() == torch::kFloat, "Input a tensor must be float!");
    TORCH_CHECK(a.is_cuda(), "Input a tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Input and grad tensors must have the same number of elements!");

    auto grad_outputs_contig = grad_outputs.contiguous(); // Apparently since the loss output might be non-contiguous

    auto grad_x = torch::empty_like(x);
    auto grad_a = torch::zeros_like(a);

    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    xsss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        a.data_ptr<float>(),
        grad_outputs_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        size
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "xsss_backward_kernel launch failed");

    return {grad_x, grad_a};
}

// ===================================================================
// AUTOGRAD CLASS DEFINITIONS

torch::Tensor xSSSAutograd::forward(AutogradContext *ctx, Tensor x, Tensor a) {
    ctx->save_for_backward({x, a});
    return forward_cuda(x, a);
}

variable_list xSSSAutograd::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto a = saved[1];

    std::vector<torch::Tensor> grads = backward_cuda(x, a, grad_outputs[0]);
    
    variable_list result(2);
    result[0] = grads[0]; // grad_x
    result[1] = grads[1]; // grad_a

    return result;
}