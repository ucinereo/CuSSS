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

// ===================================================================
// Templated element-wise operations

template <typename T> struct sssglu_forward_op {
  __device__ static T forward(T x, T y) {
    float x_f = static_cast<float>(x);
    float y_f = static_cast<float>(y);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float sss_val = (x_f * inv) * 0.5f + 0.5f;
    float result = sss_val * x_f * y_f;
    return static_cast<T>(result);
  }
};

template <> struct sssglu_forward_op<float> {
  __device__ static float forward(float x, float y) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    return ((x * inv) * 0.5f + 0.5f) * x * y;
  }
};

template <typename T> struct sssglu_backward_x_op {
  __device__ static T backward(T x, T y, T grad_output) {
    float x_f = static_cast<float>(x);
    float y_f = static_cast<float>(y);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float x_inv = x_f * inv;
    float grad_x = grad_output_f * 0.5f * y_f * (x_inv * inv + x_inv + 1.0f);
    return static_cast<T>(grad_x);
  }
};

template <> struct sssglu_backward_x_op<float> {
  __device__ static float backward(float x, float y, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float x_inv = x * inv;
    return grad_output * 0.5f * y * (x_inv * inv + x_inv + 1.0f);
  }
};

template <typename T> struct sssglu_backward_y_op {
  __device__ static T backward(T x, T y, T grad_output) {
    float x_f = static_cast<float>(x);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float x_inv = x_f * inv;
    float grad_y = grad_output_f * (x_inv * 0.5f + 0.5f) * x_f;
    return static_cast<T>(grad_y);
  }
};

template <> struct sssglu_backward_y_op<float> {
  __device__ static float backward(float x, float y, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float x_inv = x * inv;
    return grad_output * (x_inv * 0.5f + 0.5f) * x;
  }
};

// ===================================================================
// CUDA KERNELS

// Helper to apply a binary operation element-wise to two vectors
template <typename vec_t, typename native_t, int N, typename Op>
struct VectorApplyBinary;

template <typename vec_t, typename native_t, typename Op>
struct VectorApplyBinary<vec_t, native_t, 4, Op> {
  __device__ static vec_t apply(const vec_t &v1, const vec_t &v2) {
    return {Op::forward(v1.x, v2.x), Op::forward(v1.y, v2.y),
            Op::forward(v1.z, v2.z), Op::forward(v1.w, v2.w)};
  }

  __device__ static vec_t apply_backward(const vec_t &v1, const vec_t &v2, const vec_t &grad) {
    return {Op::backward(v1.x, v2.x, grad.x), Op::backward(v1.y, v2.y, grad.y),
            Op::backward(v1.z, v2.z, grad.z), Op::backward(v1.w, v2.w, grad.w)};
  }
};

template <typename vec_t, typename native_t, typename Op>
struct VectorApplyBinary<vec_t, native_t, 2, Op> {
  __device__ static vec_t apply(const vec_t &v1, const vec_t &v2) {
    return {Op::forward(v1.x, v2.x), Op::forward(v1.y, v2.y)};
  }

  __device__ static vec_t apply_backward(const vec_t &v1, const vec_t &v2, const vec_t &grad) {
    return {Op::backward(v1.x, v2.x, grad.x), Op::backward(v1.y, v2.y, grad.y)};
  }
};

template <typename vec_t, typename native_t, typename Op>
struct VectorApplyBinary<vec_t, native_t, 1, Op> {
  __device__ static vec_t apply(const vec_t &v1, const vec_t &v2) {
    return Op::forward(v1, v2);
  }

  __device__ static vec_t apply_backward(const vec_t &v1, const vec_t &v2, const vec_t &grad) {
    return Op::backward(v1, v2, grad);
  }
};

template <typename scalar_t>
__global__ void sssglu_forward_kernel(const scalar_t* x, const scalar_t* y, scalar_t* output, int size) {
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
        vec_t w = reinterpret_cast<const vec_t*>(y)[i_vec];

        // apply operation
        vec_t out = VectorApplyBinary<vec_t, native_t, vec_size, sssglu_forward_op<native_t>>::apply(v, w);

        // vectorized store
        reinterpret_cast<vec_t*>(output)[i_vec] = out;
    }
}

template <typename scalar_t>
__global__ void sssglu_backward_kernel(const scalar_t* x, const scalar_t* y, const scalar_t* grad_out, scalar_t* grad_x, scalar_t* grad_y, int size) {
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
        vec_t w = reinterpret_cast<const vec_t*>(y)[i_vec];
        vec_t g = reinterpret_cast<const vec_t*>(grad_out)[i_vec];

        // apply backward operations
        vec_t grad_x_vec = VectorApplyBinary<vec_t, native_t, vec_size, sssglu_backward_x_op<native_t>>::apply_backward(v, w, g);
        vec_t grad_y_vec = VectorApplyBinary<vec_t, native_t, vec_size, sssglu_backward_y_op<native_t>>::apply_backward(v, w, g);

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = grad_x_vec;
        reinterpret_cast<vec_t*>(grad_y)[i_vec] = grad_y_vec;
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor sssglu_forward_cuda(const at::Tensor& x_in, const at::Tensor& y_in) {
    auto x = fp8_to_float(x_in);
    SSS_DTYPE_CHECK(x, "Input 'x' tensor");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto y = fp8_to_float(y_in);
    SSS_DTYPE_CHECK(y, "Input 'y' tensor");
    TORCH_CHECK(y.is_cuda(), "Input 'y' tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");

    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "sssglu_forward_cuda", [&] {
        sssglu_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });
    output = float_to_fp8(output, x_in.scalar_type());
    return output;
}

tensor_list sssglu_backward_cuda(const at::Tensor& x_in, const at::Tensor& y_in, const at::Tensor& grad_output_in) {
    auto x = fp8_to_float(x_in);
    SSS_DTYPE_CHECK(x, "Input 'x' tensor");
    TORCH_CHECK(x.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto y = fp8_to_float(y_in);
    SSS_DTYPE_CHECK(y, "Input 'y' tensor");
    TORCH_CHECK(y.is_cuda(), "Input 'y' tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == y.numel(), "Input tensors must have the same number of elements!");
    auto grad_output = fp8_to_float(grad_output_in);
    SSS_DTYPE_CHECK(grad_output, "Grad tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_output.numel(), "Grad tensor must match x tensor size!");

    auto grad_output_contig = grad_output.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x).contiguous();
    auto grad_y = torch::empty_like(y).contiguous();
    int size = x.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "sssglu_backward_cuda", [&] {
        sssglu_backward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            size
        );
        });

    grad_x = float_to_fp8(grad_x, x_in.scalar_type());
    grad_y = float_to_fp8(grad_y, y_in.scalar_type());
    return {grad_x, grad_y};
}