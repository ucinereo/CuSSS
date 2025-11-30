#include "sss_glu_impl.hpp"

#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS
// ===================================================================

__global__ void sss_glu_forward_kernel(const float* x, const float* y, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        float o = (e * inv) * 0.5f + 0.5f;
        output[idx] = o * e * y[idx];
    }
}

__global__ void sss_glu_backward_kernel(const float* x, const float* y, const float* grad_out, float* grad_x, float* grad_y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float g = y[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        float l = e * inv;
        grad_x[idx] = 0.5f * g * (l * inv + l + 1.0f) * grad_out[idx];
        grad_y[idx] = (l * 0.5f + 0.5f) * e * grad_out[idx];
    }
}


// ===================================================================
// FORWARD AND BACKWARD IMPLEMENTATIONS
// ===================================================================

torch::Tensor forward_cuda(torch::Tensor &x, torch::Tensor &y) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor x must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor!");
    TORCH_CHECK(y.dtype() == torch::kFloat, "Input tensor y must be float!");
    TORCH_CHECK(y.is_cuda(), "Input tensor y must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");

    auto output = torch::empty_like(x);
    int size = x.numel();

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_glu_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}

std::vector<torch::Tensor> backward_cuda(torch::Tensor &x, torch::Tensor &y, torch::Tensor &grad_outputs) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor x must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor!");
    TORCH_CHECK(y.dtype() == torch::kFloat, "Input tensor y must be float!");
    TORCH_CHECK(y.is_cuda(), "Input tensor y must be a CUDA tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must have the same number of elements as inputs!");

    auto grad_outputs_contig = grad_outputs.contiguous();

    auto grad_x = torch::empty_like(x);
    auto grad_y = torch::empty_like(y);
    int size = x.numel();

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_glu_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_outputs_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        size
    );

    return {grad_x, grad_y};
}


// ===================================================================
// AUTOGRAD CLASS DEFINITIONS
// ===================================================================

torch::Tensor SSSGLUAutograd::forward(AutogradContext *ctx, Tensor x, Tensor y) {
    ctx->save_for_backward({x, y});
    return forward_cuda(x, y);
}

variable_list SSSGLUAutograd::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto y = saved[1];
    return backward_cuda(x, y, grad_outputs[0]);
}
