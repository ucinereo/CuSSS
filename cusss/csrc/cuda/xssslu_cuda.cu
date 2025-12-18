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

__global__ void xssslu_forward_kernel(const float* x, const float* a, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    if (base + 3 < size) {
        // vectorized load
        float4 vx = reinterpret_cast<const float4*>(x)[i4];

        // Explicit scalar lanes
        float e0x = vx.x;
        float e1x = vx.y;
        float e2x = vx.z;
        float e3x = vx.w; 

        float a0 = a[0];

        float o0 = e0x * ((e0x * __frcp_rn(1.0f + fabsf(e0x))) * a0 + 0.5f);
        float o1 = e1x * ((e1x * __frcp_rn(1.0f + fabsf(e1x))) * a0 + 0.5f);
        float o2 = e2x * ((e2x * __frcp_rn(1.0f + fabsf(e2x))) * a0 + 0.5f);
        float o3 = e3x * ((e3x * __frcp_rn(1.0f + fabsf(e3x))) * a0 + 0.5f);

        float4 out;
        out.x = o0;
        out.y = o1;
        out.z = o2;
        out.w = o3;

        // vectorized store
        reinterpret_cast<float4*>(output)[i4] = out;
    }
}

__global__ void xssslu_backward_kernel(const float* x, const float* a, const float* grad_out, float* grad_x, float* grad_a, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // index in units of float4
    int i4 = tid;
    int base = i4 * 4;

    // declare shared memory for block-wise reduction
    extern __shared__ float sdata[]; // allocated at launch: blockDim.x * sizeof(float)

    // compute local contribution; threads out-of-range write 0.0f so they participate correctly in reduction
    float local_a = 0.0f;

    if (base + 3 < size) {
        // vectorized load
        float4 vx = reinterpret_cast<const float4*>(x)[i4];
        float4 g = reinterpret_cast<const float4*>(grad_out)[i4]; 
        
        // Explicit scalar lanes
        float e0x = vx.x;
        float e1x = vx.y;
        float e2x = vx.z;
        float e3x = vx.w; 
        
        float a0 = a[0];

        // helper
        float inv0 = __frcp_rn(1.0f + fabsf(e0x));
        float inv1 = __frcp_rn(1.0f + fabsf(e1x));
        float inv2 = __frcp_rn(1.0f + fabsf(e2x));
        float inv3 = __frcp_rn(1.0f + fabsf(e3x));
        
        // df/dx
        float o0x = g.x * ((2 * fabsf(e0x) * (a0 * e0x + 1) + e0x * e0x + 4.0f * a0 * e0x + 1) * (inv0 * inv0 * 0.5f));
        float o1x = g.y * ((2 * fabsf(e1x) * (a0 * e1x + 1) + e1x * e1x + 4.0f * a0 * e1x + 1) * (inv1 * inv1 * 0.5f));
        float o2x = g.z * ((2 * fabsf(e2x) * (a0 * e2x + 1) + e2x * e2x + 4.0f * a0 * e2x + 1) * (inv2 * inv2 * 0.5f));
        float o3x = g.w * ((2 * fabsf(e3x) * (a0 * e3x + 1) + e3x * e3x + 4.0f * a0 * e3x + 1) * (inv3 * inv3 * 0.5f));
        
        // df/da
        float o0a = g.x * e0x * e0x * inv0;
        float o1a = g.y * e1x * e1x * inv1;
        float o2a = g.z * e2x * e2x * inv2;
        float o3a = g.w * e3x * e3x * inv3;

        // vectorized store
        float4 outx;
        outx.x = o0x;
        outx.y = o1x;
        outx.z = o2x;
        outx.w = o3x;
        reinterpret_cast<float4*>(grad_x)[i4] = outx;

        local_a = o0a + o1a + o2a + o3a;
    }

    // write local contribution into shared memory for reduction
    sdata[threadIdx.x] = local_a;
    __syncthreads();

    // reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // thread 0 performs a single atomic add to global grad_a
    if (threadIdx.x == 0) {
        atomicAdd(grad_a, sdata[0]);
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor xssslu_forward_cuda(const at::Tensor& x, const at::Tensor& a) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input 'x' tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.dtype() == torch::kFloat, "Input 'a' tensor must be float!");
    TORCH_CHECK(a.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");


    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    xssslu_forward_kernel<<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), a.data_ptr<float>(), output.data_ptr<float>(), size
    );

    return output;
}

tensor_list xssslu_backward_cuda(const at::Tensor& x, const at::Tensor& a, const at::Tensor& grad_output) {
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input 'x' tensor must be float!");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.dtype() == torch::kFloat, "Input 'a' tensor must be float!");
    TORCH_CHECK(a.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat, "Grad tensor must be float!");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    auto grad_a = torch::zeros_like(a);
    int size = x.numel();

    int blockSize = 128;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    // allocate shared memory: one float per thread for partial sums
    xssslu_backward_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>> (
        x.data_ptr<float>(),
        a.data_ptr<float>(),
        grad_output_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        size
    );

    return {grad_x, grad_a};
}
