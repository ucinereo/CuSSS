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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float e = x[idx];
        float inv = __frcp_rn(1.0f + fabsf(e));
        float grad = inv * inv * 0.5f;
        grad_x[idx] = grad_out[idx] * grad;
    }
}


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
    TORCH_CHECK(x.is_cuda() && grad_output.is_cuda(), "CUDA only");

    auto grad_output_contig = grad_output.contiguous();

    auto grad_x = at::empty_like(x);

    int size = x.numel();


    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;


    // <<<blocks, threads>>>
    sss_backward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_output_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        x.numel()
    );

    return grad_x;
}




