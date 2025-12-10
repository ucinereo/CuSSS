#include "sss_cuda.hpp"

#include <cuda.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;

// ===================================================================
// CUDA KERNELS

__global__ void sss_forward_kernel(const float* __restrict__ x, float* __restrict__ output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    // Vectorized path
    if (base + 3 < size) {
        float4 v = reinterpret_cast<const float4*>(x)[i4];

        float4 out;
        out.x = (v.x * __frcp_rn(1.0f + fabsf(v.x))) * 0.5f + 0.5f;
        out.y = (v.y * __frcp_rn(1.0f + fabsf(v.y))) * 0.5f + 0.5f;
        out.z = (v.z * __frcp_rn(1.0f + fabsf(v.z))) * 0.5f + 0.5f;
        out.w = (v.w * __frcp_rn(1.0f + fabsf(v.w))) * 0.5f + 0.5f;

        reinterpret_cast<float4*>(output)[i4] = out;
    }
    // Tail Path
    else if (base < size) {
        // Handle remaining 1, 2, or 3 elements
        for (int k = 0; k < 4; ++k) {
            int idx = base + k;
            if (idx < size) {
                float val = x[idx];
                output[idx] = (val * __frcp_rn(1.0f + fabsf(val))) * 0.5f + 0.5f;
            }
        }
    }
}

__global__ void sss_backward_kernel(const float* __restrict__ y, const float* __restrict__ grad_out, float* __restrict__ grad_x, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    // Vectorized path
    if (base + 3 < size) {
        float4 v = reinterpret_cast<const float4*>(y)[i4];
        float4 g = reinterpret_cast<const float4*>(grad_out)[i4];

        float t0 = 1.0f - fabsf(2.0f * v.x - 1.0f);
        float t1 = 1.0f - fabsf(2.0f * v.y - 1.0f);
        float t2 = 1.0f - fabsf(2.0f * v.z - 1.0f);
        float t3 = 1.0f - fabsf(2.0f * v.w - 1.0f);

        float4 out;
        out.x = g.x * (0.5f * t0 * t0);
        out.y = g.y * (0.5f * t1 * t1);
        out.z = g.z * (0.5f * t2 * t2);
        out.w = g.w * (0.5f * t3 * t3);

        reinterpret_cast<float4*>(grad_x)[i4] = out;
    }
    // Tail Path
    else if (base < size) {
        for (int k = 0; k < 4; ++k) {
            int idx = base + k;
            if (idx < size) {
                float val = y[idx];
                float g_val = grad_out[idx];

                float t = 1.0f - fabsf(2.0f * val - 1.0f);
                grad_x[idx] = g_val * (0.5f * t * t);
            }
        }
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor sss_forward_cuda(const at::Tensor& x) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);
    int size = x_contig.numel();

    int blockSize = 1024;
    // Round up to ensure tail is covered
    int num_vectors = (size + 3) / 4;
    int numBlocks = (num_vectors + blockSize - 1) / blockSize;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sss_forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

at::Tensor sss_backward_cuda(const at::Tensor& y, const at::Tensor& grad_output) {
    TORCH_CHECK(y.dtype() == torch::kFloat, "Input tensor must be float!");
    TORCH_CHECK(y.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");

    // Ensure contiguity for float4 alignment
    auto y_contig = y.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    // Create output buffer matching the contiguous y
    auto grad_x = torch::empty_like(y_contig);
    int size = y_contig.numel();

    int blockSize = 1024;
    // Round up to ensure tail is covered
    int num_vectors = (size + 3) / 4;
    int numBlocks = (num_vectors + blockSize - 1) / blockSize;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sss_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        y_contig.data_ptr<float>(),
        grad_output_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    return grad_x;
}
