#include "sss_cuda.hpp"
#include "../utils/template_utils.hpp"
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;

// ===================================================================
// Templated element-wise operations

template <typename T> struct sss_elementwise_op {
  __device__ static T forward(T x) {
    float x_f = static_cast<float>(x);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float result = (x_f * inv) * 0.5f + 0.5f;
    return static_cast<T>(result);
  }
  __device__ static T backward(T x, T grad_output) {
    float x_f = static_cast<float>(x);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float grad_input = grad_output_f * 0.5f * inv * inv;
    return static_cast<T>(grad_input);
  }
};

template <> struct sss_elementwise_op<float> {
  __device__ static float forward(float x) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    return (x * inv) * 0.5f + 0.5f;
  }

  __device__ static float backward(float x, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float grad_input = grad_output * 0.5f * inv * inv;
    return grad_input;
  }
};

// ===================================================================
// CUDA Kernels

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

at::Tensor sss_forward_cuda(const at::Tensor& x) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x_, "Input tensor");
    TORCH_CHECK(x_.is_cuda(), "Input tensor must be a CUDA tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x_).contiguous();
    int size = x_.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x_.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::kBFloat16, x_.scalar_type(), "sss_forward_cuda", [&] {
        sss_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });
    output = float_to_fp8(output, x_.scalar_type());
    return output;
}

at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output_in) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x_, "Input tensor");  
    TORCH_CHECK(x_.is_cuda(), "Input tensor must be a CUDA tensor!");
    auto grad_output = fp8_to_float(grad_output_in);
    SSS_DTYPE_CHECK(grad_output, "Grad tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    // Ensure contiguity for float4 alignment
    auto x_contig = x_.contiguous();
    auto grad_output_contig = grad_output.contiguous();

    // Create output buffer matching the contiguous x
    auto grad_x = torch::empty_like(x_contig);
    int size = x_contig.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x_.scalar_type());
    int num_vec = (size + vec_size - 1) / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, x_.scalar_type(), "sss_backward_cuda", [&] {
        sss_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x_.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            size
        );
      });
    grad_x = float_to_fp8(grad_x, x_.scalar_type());
    return grad_x;
}
