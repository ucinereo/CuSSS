#include "sss_impl.hpp"
#include "../../utils/template_utils.hpp"
#include <cuda_bf16.h>

#include <cuda.h>
#include <torch/script.h>
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

template <typename scalar_t>
__global__ void sss_forward_kernel(const scalar_t* x, scalar_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t e = x[idx];
        output[idx] = sss_elementwise_op<scalar_t>::forward(e);
    }
}

template <typename scalar_t>
__global__ void sss_backward_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t e = x[idx];
        grad_x[idx] = sss_elementwise_op<scalar_t>::backward(e, grad_out[idx]);
    }
}

__global__ void sss_forward_kernel_f4(const float* x, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    if (base + 3 < size) {
        // vectorized load
        float4 v = reinterpret_cast<const float4*>(x)[i4];

        // Explicit scalar lanes
        float e0 = v.x;
        float e1 = v.y;
        float e2 = v.z;
        float e3 = v.w;

        // elementwise SSS computation
        float o0 = (e0 * __frcp_rn(1.0f + fabsf(e0))) * 0.5f + 0.5f;
        float o1 = (e1 * __frcp_rn(1.0f + fabsf(e1))) * 0.5f + 0.5f;
        float o2 = (e2 * __frcp_rn(1.0f + fabsf(e2))) * 0.5f + 0.5f;
        float o3 = (e3 * __frcp_rn(1.0f + fabsf(e3))) * 0.5f + 0.5f;

        // pack results
        float4 out;
        out.x = o0;
        out.y = o1;
        out.z = o2;
        out.w = o3;

        // vectorized store
        reinterpret_cast<float4*>(output)[i4] = out;
    }
}

__global__ void sss_forward_tail_kernel(const float* x, float* output, int start, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        output[idx] = (e * inv) * 0.5f + 0.5f;
    }
}

__global__ void sss_backward_kernel_f4(const float* x, const float* grad_out, float* grad_x, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    if (base + 3 < size) {
        // vectorized load
        float4 v = reinterpret_cast<const float4*>(x)[i4];
        float4 g = reinterpret_cast<const float4*>(grad_out)[i4];

        // Explicit scalar lanes
        float e0 = v.x;
        float e1 = v.y;
        float e2 = v.z;
        float e3 = v.w;

        // elementwise SSS computation
        float inv0 = __frcp_rn(1.0f + fabsf(e0));
        float o0 = (inv0 * inv0) * 0.5f;
        float inv1 = __frcp_rn(1.0f + fabsf(e1));
        float o1 = (inv1 * inv1) * 0.5f;
        float inv2 = __frcp_rn(1.0f + fabsf(e2));
        float o2 = (inv2 * inv2) * 0.5f;
        float inv3 = __frcp_rn(1.0f + fabsf(e3));
        float o3 = (inv3 * inv3) * 0.5f;

        // pack results
        float4 out;
        out.x = g.x * o0;
        out.y = g.y * o1;
        out.z = g.z * o2;
        out.w = g.w * o3;

        // vectorized store
        reinterpret_cast<float4*>(grad_x)[i4] = out;
    } 
}

__global__ void sss_backward_tail_kernel(const float* x, const float* grad_out, float* grad_x, int start, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
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
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must have the same number of elements as input tensor!");

    auto grad_outputs_contig = grad_outputs.contiguous(); // Apparently since the loss output might be non-contiguous

    auto grad_x = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_outputs_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    return {grad_x};
}

torch::Tensor forward_cuda_f4(torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sss_forward_kernel_f4<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), size
    );

    int tail = size % 4; // remaining elements
    if (tail > 0) {
        int tailBlockSize = 32;  // small, enough for 1–3 elements
        int tailNumBlocks = (tail + tailBlockSize - 1) / tailBlockSize;

        sss_forward_tail_kernel<<<tailNumBlocks, tailBlockSize>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), num4 * 4, size
        );
    }

    return output;
}

std::vector<torch::Tensor> backward_cuda_f4(torch::Tensor &x, torch::Tensor &grad_outputs) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must be a CUDA tensor!");

    grad_outputs = grad_outputs.contiguous(); // Apparently since the loss output might be non-contiguous
    x = x.contiguous();
    auto grad_x = torch::empty_like(x).contiguous();
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sss_backward_kernel_f4<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_outputs.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    int tail = size % 4; // remaining elements
    if (tail > 0) {
        int tailBlockSize = 32;  // small, enough for 1–3 elements
        int tailNumBlocks = (tail + tailBlockSize - 1) / tailBlockSize;

        sss_backward_tail_kernel<<<tailNumBlocks, tailBlockSize>>>(
            x.data_ptr<float>(), 
            grad_outputs.data_ptr<float>(),
            grad_x.data_ptr<float>(), 
            num4 * 4, 
            size
        );
    }

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

torch::Tensor SSSAutograd_f4::forward(AutogradContext *ctx, Tensor x) {
    ctx->save_for_backward({x});
    return forward_cuda_f4(x);
}

variable_list SSSAutograd_f4::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    return backward_cuda_f4(x, grad_outputs[0]);
}