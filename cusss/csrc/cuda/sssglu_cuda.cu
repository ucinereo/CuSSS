#include "sss_cuda.hpp"

#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;
using torch::autograd::tensor_list;


// ===================================================================
// CUDA KERNELS

__global__ void sssglu_forward_kernel(const float* x, const float* y, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    if (base + 3 < size) {
        // vectorized load
        float4 v = reinterpret_cast<const float4*>(x)[i4];
        float4 w = reinterpret_cast<const float4*>(y)[i4];

        // Explicit scalar lanes
        float e0x = v.x;
        float e1x = v.y;
        float e2x = v.z;
        float e3x = v.w;

        float e0y = w.x;
        float e1y = w.y;
        float e2y = w.z;
        float e3y = w.w;

        // elementwise SSSGLU computation
        float o0 = ((e0x * __frcp_rn(1.0f + fabsf(e0x))) * 0.5f + 0.5f) * e0x * e0y;
        float o1 = ((e1x * __frcp_rn(1.0f + fabsf(e1x))) * 0.5f + 0.5f) * e1x * e1y;
        float o2 = ((e2x * __frcp_rn(1.0f + fabsf(e2x))) * 0.5f + 0.5f) * e2x * e2y;
        float o3 = ((e3x * __frcp_rn(1.0f + fabsf(e3x))) * 0.5f + 0.5f) * e3x * e3y;

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

__global__ void sssglu_backward_kernel(const float* x, const float* y, const float* grad_out, float* grad_x, float* grad_y, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    if (base + 3 < size) {
        // vectorized load
        float4 v = reinterpret_cast<const float4*>(x)[i4];
        float4 w = reinterpret_cast<const float4*>(y)[i4];
        float4 g = reinterpret_cast<const float4*>(grad_out)[i4];

        // Explicit scalar lanes
        float e0x = v.x;
        float e1x = v.y;
        float e2x = v.z;
        float e3x = v.w;

        float e0y = w.x;
        float e1y = w.y;
        float e2y = w.z;
        float e3y = w.w;

        // elementwise SSS computation
        float inv0 = __frcp_rn(1.0f + fabsf(e0x));
        float inv1 = __frcp_rn(1.0f + fabsf(e1x));
        float inv2 = __frcp_rn(1.0f + fabsf(e2x));
        float inv3 = __frcp_rn(1.0f + fabsf(e3x));

        float x_inv0 = e0x * inv0;
        float x_inv1 = e1x * inv1;
        float x_inv2 = e2x * inv2;
        float x_inv3 = e3x * inv3;

        float o0x = 0.5f * e0y * (x_inv0 * inv0 + x_inv0 + 1.0f);
        float o1x = 0.5f * e1y * (x_inv1 * inv1 + x_inv1 + 1.0f);
        float o2x = 0.5f * e2y * (x_inv2 * inv2 + x_inv2 + 1.0f);
        float o3x = 0.5f * e3y * (x_inv3 * inv3 + x_inv3 + 1.0f);

        float o0y = (x_inv0 * 0.5f + 0.5f) * e0x;
        float o1y = (x_inv1 * 0.5f + 0.5f) * e1x;
        float o2y = (x_inv2 * 0.5f + 0.5f) * e2x;
        float o3y = (x_inv3 * 0.5f + 0.5f) * e3x;

        // pack results
        float4 outx;
        outx.x = g.x * o0x;
        outx.y = g.y * o1x;
        outx.z = g.z * o2x;
        outx.w = g.w * o3x;

        float4 outy;
        outy.x = g.x * o0y;
        outy.y = g.y * o1y;
        outy.z = g.z * o2y;
        outy.w = g.w * o3y;

        // vectorized store
        reinterpret_cast<float4*>(grad_x)[i4] = outx;
        reinterpret_cast<float4*>(grad_y)[i4] = outy;
    } 
}

// ===================================================================
// Kernel Launchers

at::Tensor sssglu_forward_cuda(const at::Tensor& x, const at::Tensor& y) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input 'x' tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    TORCH_CHECK(y.dtype() == torch::kFloat, "Input 'y' tensor must be float!");
    TORCH_CHECK(y.is_cuda(), "Input 'y' tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");

    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sssglu_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}

tensor_list sssglu_backward_cuda(const at::Tensor& x, const at::Tensor& y, const at::Tensor& grad_output) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input 'x' tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    TORCH_CHECK(y.dtype() == torch::kFloat, "Input 'y' tensor must be float!");
    TORCH_CHECK(y.is_cuda(), "Input 'y' tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    auto grad_y = torch::empty_like(y).contiguous();
    int size = x.numel();

    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sssglu_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_output_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        size
    );

    return {grad_x, grad_y};
}