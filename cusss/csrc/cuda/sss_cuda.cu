#include "sss_cuda.hpp"

#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

__global__ void sss_forward_kernel(const float* x, float* output, int size) {
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

__global__ void sss_backward_kernel(const float* x, const float* grad_out, float* grad_x, int size) {
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

// ===================================================================
// Kernel Launchers

at::Tensor sss_forward_cuda(const at::Tensor& x) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sss_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}

at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    int size = x.numel();

    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_output_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    return grad_x;
}
