#include "ssslu_cuda.hpp"
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

template <typename T> struct ssslu_elementwise_op {
  __device__ static T forward(T x) {
    float x_f = static_cast<float>(x);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float result = x_f * ((x_f * inv) * 0.5f + 0.5f);
    return static_cast<T>(result);
  }
  __device__ static T backward(T x, T grad_output) {
    float x_f = static_cast<float>(x);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float t = x_f + 1;
    float o = ((x_f + 2) * fabsf(x_f) + (t * t)) * ((inv * inv) * 0.5f);
    float grad_input = grad_output_f * o;
    return static_cast<T>(grad_input);
  }
};

template <> struct ssslu_elementwise_op<float> {
  __device__ static float forward(float x) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    return x * ((x * inv) * 0.5f + 0.5f);
  }

  __device__ static float backward(float x, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float t = x + 1;
    float o = ((x + 2) * fabsf(x) + (t * t)) * ((inv * inv) * 0.5f);
    float grad_input = grad_output * o;
    return grad_input;
  }
};

template <typename scalar_t>
__global__ void ssslu_forward_kernel(const scalar_t* x, scalar_t* output, int size) {
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
        vec_t out = Traits::template apply<ssslu_elementwise_op<native_t>>(v);

        // vectorized store
        reinterpret_cast<vec_t*>(output)[i_vec] = out;
    }
}

template <typename scalar_t>
__global__ void ssslu_backward_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
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
        vec_t out = Traits::template apply_backward<ssslu_elementwise_op<native_t>>(v, g);

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = out;
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor ssslu_forward_cuda(const at::Tensor& x) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x, "Input tensor");
    TORCH_CHECK(x_.is_cuda(), "Input tensor must be a CUDA tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x_).contiguous();
    int size = x_.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::kBFloat16, x_.scalar_type(), "ssslu_forward_cuda", [&] {
        ssslu_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });

    return output;
}

at::Tensor ssslu_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x, "Input tensor");
    auto grad_output_ = fp8_to_float(grad_output);
    TORCH_CHECK(x_.is_cuda(), "Input tensor must be a CUDA tensor!");
    SSS_DTYPE_CHECK(grad_output, "Grad tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x_.numel() == grad_output.numel(), "Grad tensor must be a CUDA tensor!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x_).contiguous();
    int size = x_.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x_.scalar_type());
    int num_vec = size/vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, x_.scalar_type(), "ssslu_backward_cuda", [&] {
        ssslu_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x_.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            size
        );
      });
    grad_x = float_to_fp8(grad_x, x.scalar_type());
    return grad_x;
}
