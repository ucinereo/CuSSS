#include "sss_cuda.hpp"
#include "../utils/template_utils.hpp"

#include <cuda_bf16.h>
#include <cuda.h>
#include <torch/script.h>

using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;
using torch::autograd::tensor_list;


// ===================================================================
// CUDA KERNELS

template <typename T> struct xssslu_elementwise_op {
  __device__ static T forward(T x, T a) {
    float x_f = static_cast<float>(x);
    float a_f = static_cast<float>(a);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float result = x_f * ((x_f * inv) * a_f + 0.5f);
    return static_cast<T>(result);
  }

  __device__ static T backward_x(T x, T a, T grad_output) {
    float x_f = static_cast<float>(x);
    float a_f = static_cast<float>(a);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float o = ((2 * fabsf(x_f) * (a_f * x_f + 1) + x_f * x_f + 4.0f * a_f * x_f + 1) * (inv * inv * 0.5f));
    float grad_input = grad_output_f * o;
    return static_cast<T>(grad_input);
  }

  __device__ static T backward_a(T x, T grad_output) {
    float x_f = static_cast<float>(x);
    float grad_output_f = static_cast<float>(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float o = x_f * x_f * inv;
    float grad_a = grad_output_f * x_f * inv;
    return static_cast<T>(grad_a);
  }
};

template <> struct xssslu_elementwise_op<float> {
  __device__ static float forward(float x, float a) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    return x * ((x * inv) * a + 0.5f);
  }

  __device__ static float backward_x(float x, float a, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float o = ((2 * fabsf(x) * (a * x + 1) + x * x + 4.0f * a * x + 1) * (inv * inv * 0.5f));
    return grad_output * o;
  }

  __device__ static float backward_a(float x, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float o = x * x * inv;
    return grad_output * o;
  }
};

template <typename scalar_t>
__global__ void xssslu_forward_kernel(const scalar_t* x, const scalar_t* a, scalar_t* output, int size) {
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
        vec_t out = Traits::template apply_xssslu<xssslu_elementwise_op<native_t>>(v, a_val);

        // vectorized store
        reinterpret_cast<vec_t*>(output)[i_vec] = out;
    }
}

template <typename scalar_t>
__global__ void xssslu_backward_kernel(const scalar_t* x, const scalar_t* a, const scalar_t* grad_out, scalar_t* grad_x, scalar_t* grad_a, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    using Traits = VectorIO<scalar_t>;
    using vec_t = typename Traits::vec_t;
    using native_t = typename Traits::native_t;
    constexpr int vec_size = Traits::packed_size;

    int i_vec = tid;
    int base = i_vec * vec_size;

    // declare shared memory for block-wise reduction
    extern __shared__ float sdata[]; // allocated at launch: blockDim.x * sizeof(float)

    // compute local contribution; threads out-of-range write 0.0f so they participate correctly in reduction
    float local_a = 0.0f;

    if (base + vec_size - 1 < size) {
        // vectorized load
        vec_t v = reinterpret_cast<const vec_t*>(x)[i_vec];
        vec_t g = reinterpret_cast<const vec_t*>(grad_out)[i_vec];
        native_t a_val = to_native<scalar_t, native_t>(a[0]);
        // apply backward operation for x
        vec_t grad_x_vec = Traits::template apply_backward_x<xssslu_elementwise_op<native_t>>(v, a_val, g);

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = grad_x_vec;

        // accumulate gradient for a
        local_a = static_cast<float>(Traits::template apply_backward_a<xssslu_elementwise_op<native_t>>(v, g));
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
      native_t* grad_a_ptr = reinterpret_cast<native_t*>(grad_a);
      atomicAdd(grad_a_ptr, static_cast<native_t>(sdata[0]));
    }
}

// ===================================================================
// Kernel Launchers

at::Tensor xssslu_forward_cuda(const at::Tensor& x, const at::Tensor& a) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x_, "Input 'x' tensor");
    TORCH_CHECK(x_.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto a_ = fp8_to_float(a);
    SSS_DTYPE_CHECK(a_, "Input 'a' tensor");
    TORCH_CHECK(a_.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a_.numel() == 1, "Input a must be a scalar tensor!");

    // x = x.contiguous();
    auto output = torch::empty_like(x_).contiguous();
    int size = x_.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = get_vector_size(x_.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x_.scalar_type(), "xssslu_forward_cuda", [&] {
        xssslu_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
            x_.data_ptr<scalar_t>(),
            a_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
        });
    output = float_to_fp8(output, x_.scalar_type());
    return output;
}

tensor_list xssslu_backward_cuda(const at::Tensor& x, const at::Tensor& a, const at::Tensor& grad_output) {
    auto x_ = fp8_to_float(x);
    SSS_DTYPE_CHECK(x_, "Input 'x' tensor");
    TORCH_CHECK(x_.is_cuda(), "Input 'x' tensor must be a CUDA tensor!");
    auto a_ = fp8_to_float(a);
    SSS_DTYPE_CHECK(a_, "Input 'a' tensor");
    TORCH_CHECK(a_.is_cuda(), "Input 'a' tensor must be a CUDA tensor!");
    TORCH_CHECK(a_.numel() == 1, "Input a must be a scalar tensor!");
    auto grad_output_ = fp8_to_float(grad_output);
    SSS_DTYPE_CHECK(grad_output_, "Grad tensor");
    TORCH_CHECK(grad_output_.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x_.numel() == grad_output_.numel(), "Grad tensor must match x tensor size!");

    auto grad_output_contig = grad_output_.contiguous(); // Apparently since the loss output might be non-contiguous
    auto grad_x = torch::empty_like(x_).contiguous();
    auto grad_a = torch::zeros_like(a_);
    int size = x_.numel();

    int blockSize = 128;
    int vec_size = get_vector_size(x_.scalar_type());
    int num_vec = size / vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;
    size_t shared_mem_size = blockSize * sizeof(float);

    // allocate shared memory: one float per thread for partial sums
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x_.scalar_type(), "xssslu_backward_cuda", [&] {
        xssslu_backward_kernel<scalar_t><<<numBlocks, blockSize, shared_mem_size>>>(
            x_.data_ptr<scalar_t>(),
            a_.data_ptr<scalar_t>(),
            grad_output_contig.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_a.data_ptr<scalar_t>(),
            size
        );
        });

    grad_x = float_to_fp8(grad_x, x_.scalar_type());
    grad_a = float_to_fp8(grad_a, a_.scalar_type());
    return {grad_x, grad_a};
}
