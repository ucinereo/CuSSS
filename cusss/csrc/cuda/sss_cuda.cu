#include "sss_cuda.hpp"
#include "../utils/template_utils.hpp"
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

template <typename scalar_t>
__global__ void sss_forward_kernel(const scalar_t* x, scalar_t* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using Traits = VectorIO<scalar_t>;
    using vec_t = typename Traits::vec_t;
    using native_t = typename Traits::native_t;
    constexpr int vec_size = Traits::packed_size;

    // index in units of float4
    int i_vec = tid;
    int base = i_vec * vec_size;

    if (base + vec_size - 1 < size) {
        // vectorized load
        vec_t v = reinterpret_cast<const vec_t*>(x)[i_vec];

        // Explicit scalar lanes
        native_t e0 = v.x;
        native_t e1 = v.y;
        native_t o0 = sss_elementwise_op<native_t>::forward(e0);
        native_t o1 = sss_elementwise_op<native_t>::forward(e1);
        native_t e2, e3, o2, o3;
        if constexpr (vec_size >= 3) {
            e2 = v.z;
            e3 = v.w;
            o2 = sss_elementwise_op<native_t>::forward(e2);
            o3 = sss_elementwise_op<native_t>::forward(e3);
        }

        // pack results
        vec_t out;
        out.x = o0;
        out.y = o1;
        if constexpr (vec_size >= 3) {
            out.z = o2;
            out.w = o3;
        }
        // vectorized store
        reinterpret_cast<vec_t*>(output)[i_vec] = out;
    }
}

template <typename scalar_t>
__global__ void sss_backward_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using Traits = VectorIO<scalar_t>;
    using vec_t = typename Traits::vec_t;
    using native_t = typename Traits::native_t;
    constexpr int vec_size = Traits::packed_size;
    // index in units of vec_t
    int i_vec = tid;
    int base = i_vec * vec_size;

    if (base + vec_size - 1 < size) {
        // vectorized load
        vec_t v = reinterpret_cast<const vec_t*>(x)[i_vec];
        vec_t g = reinterpret_cast<const vec_t*>(grad_out)[i_vec];

        // Explicit scalar lanes
        native_t e0 = v.x;
        native_t e1 = v.y;
        native_t o0 = sss_elementwise_op<native_t>::backward(e0, g.x);
        native_t o1 = sss_elementwise_op<native_t>::backward(e1, g.y);
        native_t e2, e3, o2, o3;
        if constexpr (vec_size >= 3) {
            e2 = v.z;
            e3 = v.w;
            o2 = sss_elementwise_op<native_t>::backward(e2, g.z);
            o3 = sss_elementwise_op<native_t>::backward(e3, g.w);
        }

        // pack results
        vec_t out;
        out.x = o0;
        out.y = o1;
        if constexpr (vec_size >= 3) {
            out.z = o2;
            out.w = o3;
        }

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = out;
    } 
}


// ===================================================================
// Kernel Launchers

at::Tensor sss_forward_cuda(const at::Tensor& x) {
    TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size/vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "sss_forward_cuda", [&] {
        sss_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
      });

    return output;
}

at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output) {
    TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");    
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat || grad_output.dtype() == torch::kBFloat16,
        "Grad tensor must be float or bfloat16!");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    int size = x.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size/vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "sss_backward_cuda", [&] {
        sss_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            size
        );
      });

    return grad_x;
}
