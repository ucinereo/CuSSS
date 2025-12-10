#include "sss_cuda.hpp"
#include "../utils/template_utils.hpp"
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;

#define SSS_DTYPE_CHECK(tensor, name) \
    TORCH_CHECK(tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat8_e5m2, \
        name " must be float or bfloat16!")
// ===================================================================
// CUDA KERNELS

template <typename scalar_t>
__global__ void sss_forward_kernel(const scalar_t* x, scalar_t* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using Traits = VectorIO<scalar_t>;
    using vec_t = typename Traits::vec_t;
    using native_t = typename Traits::native_t;
    constexpr int vec_size = Traits::packed_size;

    int i_vec = tid;
    int base = i_vec * vec_size;

    if (base + vec_size - 1 < size) {
        // vectorized load
        vec_t v = reinterpret_cast<const vec_t*>(x)[i_vec];

        // apply operation
        vec_t out = Traits::template apply<sss_elementwise_op<native_t>>(v);

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

    int i_vec = tid;
    int base = i_vec * vec_size;

    if (base + vec_size - 1 < size) {
        // vectorized load
        vec_t v = reinterpret_cast<const vec_t*>(x)[i_vec];
        vec_t g = reinterpret_cast<const vec_t*>(grad_out)[i_vec];

        // apply backward operation
        vec_t out = Traits::template apply_backward<sss_elementwise_op<native_t>>(v, g);

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = out;
    }
}


// ===================================================================
// Kernel Launchers

at::Tensor sss_forward_cuda(const at::Tensor& x_in) {
    auto x = fp8_to_float(x_in);
    SSS_DTYPE_CHECK(x, "Input tensor");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "sss_forward_cuda", [&] {
        sss_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });
    output = float_to_fp8(output, x.scalar_type());
    return output;
}

at::Tensor sss_backward_cuda(const at::Tensor& x_in, const at::Tensor& grad_output_in) {
    auto x = fp8_to_float(x_in);
    SSS_DTYPE_CHECK(x, "Input tensor");  
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    auto grad_output = fp8_to_float(grad_output_in);
    SSS_DTYPE_CHECK(grad_output, "Grad tensor");
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
    grad_x = float_to_fp8(grad_x, x.scalar_type());
    return grad_x;
}
