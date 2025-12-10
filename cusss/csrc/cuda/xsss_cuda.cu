#include "sss_cuda.hpp"
#include "../utils/template_utils.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;
using torch::autograd::tensor_list;

#define XSSS_DTYPE_CHECK(tensor, name) \
    TORCH_CHECK(tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat8_e5m2, \
        name " must be float or bfloat16!")

// ===================================================================
// CUDA KERNELS

template <typename scalar_t>
__global__ void xsss_forward_kernel(const scalar_t* x, const scalar_t* a, scalar_t* output, int size) {
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
        native_t a_val = to_native<scalar_t, native_t>(a[0]);

        // apply operation
        vec_t out = Traits::template apply_xsss<xsss_elementwise_op<native_t>>(v, a_val);

        // vectorized store
        reinterpret_cast<vec_t*>(output)[i_vec] = out;
    }
}

template <typename scalar_t>
__global__ void xsss_backward_kernel(const scalar_t* x, const scalar_t* a, const scalar_t* grad_out, scalar_t* grad_x, scalar_t* grad_a, int size) {
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
        native_t a_val = to_native<scalar_t, native_t>(a[0]);
        native_t* grad_a_native = reinterpret_cast<native_t*>(grad_a);
        // apply backward operation for x
        vec_t grad_x_vec = Traits::template apply_backward_x<xsss_elementwise_op<native_t>>(v, a_val, g);

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = grad_x_vec;

        // accumulate gradient for a
        native_t grad_a_val = Traits::template apply_backward_a<xsss_elementwise_op<native_t>>(v, g);
        atomicAdd(grad_a_native, grad_a_val);
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor xsss_forward_cuda(const at::Tensor& x_in, const at::Tensor& a_in) {
    auto x = fp8_to_float(x_in);
    XSSS_DTYPE_CHECK(x, "Input 'x' tensor");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto a = fp8_to_float(a_in);
    XSSS_DTYPE_CHECK(a, "Input 'a' tensor");
    TORCH_CHECK(a.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "xsss_forward_cuda", [&] {
        xsss_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });
    output = float_to_fp8(output, x_in.scalar_type());
    return output;
}

tensor_list xsss_backward_cuda(const at::Tensor& x_in, const at::Tensor& a_in, const at::Tensor& grad_output_in) {
    auto x = fp8_to_float(x_in);
    XSSS_DTYPE_CHECK(x, "Input 'x' tensor");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto a = fp8_to_float(a_in);
    XSSS_DTYPE_CHECK(a, "Input 'a' tensor");
    TORCH_CHECK(a.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a.numel() == 1, "Input a must be a scalar tensor!");
    auto grad_output = fp8_to_float(grad_output_in);
    XSSS_DTYPE_CHECK(grad_output, "Grad tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must match x tensor size!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    auto grad_a = torch::zeros_like(a);
    int size = x.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "xsss_backward_cuda", [&] {
        xsss_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_a.data_ptr<scalar_t>(),
            size
        );
      });
    grad_x = float_to_fp8(grad_x, x_in.scalar_type());
    grad_a = float_to_fp8(grad_a, a_in.scalar_type());
    return {grad_x, grad_a};
}
