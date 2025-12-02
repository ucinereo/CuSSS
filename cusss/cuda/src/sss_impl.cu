#include "sss_impl.hpp"
#include "../../utils/template_utils.hpp"
#include <cuda_bf16.h>

#include <cuda.h>
#include <torch/script.h>
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;


// ===================================================================
// CUDA KERNELS

// template <typename scalar_t>
// __global__ void sss_forward_kernel_conv(const scalar_t* x, scalar_t* output, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         scalar_t e = x[idx];
//         output[idx] = sss_elementwise_bf16_with_conv::forward(e);
//     }
// }

// template <typename scalar_t>
// __global__ void sss_backward_kernel_conv(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         scalar_t e = x[idx];
//         grad_x[idx] = sss_elementwise_bf16_with_conv::backward(e, grad_out[idx]);
//     }
// }

template <typename scalar_t>
__global__ void sss_forward_kernel(const scalar_t* x, scalar_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t e = x[idx];
        output[idx] = sss_elementwise_op<scalar_t>::forward(e);
    }
}

template <typename scalar_t>
__global__ void sss_backward_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t e = x[idx];
        grad_x[idx] = sss_elementwise_op<scalar_t>::backward(e, grad_out[idx]);
    }
}

// Explicit template instantiations
template __global__ void sss_forward_kernel<float>(const float* x, float* output, int size);
template __global__ void sss_backward_kernel<float>(const float* x, const float* grad_out, float* grad_x, int size);
template __global__ void sss_forward_kernel<c10::BFloat16>(const c10::BFloat16* x, c10::BFloat16* output, int size);
template __global__ void sss_backward_kernel<c10::BFloat16>(const c10::BFloat16* x, const c10::BFloat16* grad_out, c10::BFloat16* grad_x, int size);
// template __global__ void sss_forward_kernel_conv<c10::BFloat16>(const c10::BFloat16* x, c10::BFloat16* output, int size);
// template __global__ void sss_backward_kernel_conv<c10::BFloat16>(const c10::BFloat16* x, const c10::BFloat16* grad_out, c10::BFloat16* grad_x, int size);


template <typename scalar_t>
__global__ void sss_forward_kernel_vec(const scalar_t* x, scalar_t* output, int size) {
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
__global__ void sss_forward_tail_kernel(const scalar_t* x, scalar_t* output, int start, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        scalar_t e = x[idx];
        output[idx] = sss_elementwise_op<scalar_t>::forward(e);
    }
}

template <typename scalar_t>
__global__ void sss_backward_kernel_vec(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size) {
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
        out.x = g.x * o0;
        out.y = g.y * o1;
        if constexpr (vec_size >= 3) {
            out.z = g.z * o2;
            out.w = g.w * o3;
        }

        // vectorized store
        reinterpret_cast<vec_t*>(grad_x)[i_vec] = out;
    } 
}

template <typename scalar_t>
__global__ void sss_backward_tail_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int start, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        scalar_t e = x[idx];
        grad_x[idx] = sss_elementwise_op<scalar_t>::backward(e, grad_out[idx]);
    }
}

// ===================================================================
// FORWARD AND BACKWARD IMPLEMENTATIONS

torch::Tensor forward_cuda(torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    auto output = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "sss_forward_cuda", [&] {
          sss_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
              x.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              size);
      });

    return output;
}

// explicit instantiation for vectorized kernels
template __global__ void sss_forward_kernel_vec<float>(const float* x, float* output, int size);
template __global__ void sss_backward_kernel_vec<float>(const float* x, const float* grad_out, float* grad_x, int size);
template __global__ void sss_forward_kernel_vec<c10::BFloat16>(const c10::BFloat16* x, c10::BFloat16* output, int size);
template __global__ void sss_backward_kernel_vec<c10::BFloat16>(const c10::BFloat16* x, const c10::BFloat16* grad_out, c10::BFloat16* grad_x, int size);


std::vector<torch::Tensor> backward_cuda(torch::Tensor &x, torch::Tensor &grad_outputs) {
    TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat || grad_outputs.dtype() == torch::kBFloat16,
        "Grad tensor must be float or bfloat16!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must have the same number of elements as input tensor!");

    auto grad_outputs_contig = grad_outputs.contiguous(); // Apparently since the loss output might be non-contiguous

    auto grad_x = torch::empty_like(x);
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    sss_backward_kernel<float><<<numBlocks, blockSize>>>(
        x.data_ptr<float>(),
        grad_outputs_contig.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    return {grad_x};
}

torch::Tensor forward_cuda_vec(torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");

    x = x.contiguous();
    auto output = torch::empty_like(x).contiguous();
    int size = x.numel();
    
    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int vec_size = vec_size(x.dtype().toScalarType());
    int num_vec = size/vec_size;
    int numBlocks = (num_vec + blockSize - 1) / blockSize;
    sss_forward_kernel_vec<float><<<numBlocks, blockSize>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), size
    );

    int tail = size % vec_size; // remaining elements
    if (tail > 0) {
        int tailBlockSize = 32;  // small, enough for 1–3 elements
        int tailNumBlocks = (tail + tailBlockSize - 1) / tailBlockSize;

        sss_forward_tail_kernel<<<tailNumBlocks, tailBlockSize>>>(
            x.data_ptr<float>(), output.data_ptr<float>(), num_vec * vec_size, size
        );
    }

    return output;
}

std::vector<torch::Tensor> backward_cuda_vec(torch::Tensor &x, torch::Tensor &grad_outputs) {
        TORCH_CHECK(x.dtype() == torch::kFloat || x.dtype() == torch::kBFloat16,
        "Input tensor must be float or bfloat16!");
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor!");
    TORCH_CHECK(grad_outputs.dtype() == torch::kFloat || grad_outputs.dtype() == torch::kBFloat16,
        "Grad tensor must be float or bfloat16!");
    TORCH_CHECK(grad_outputs.is_cuda(), "Grad tensor must be a CUDA tensor!");
    TORCH_CHECK(x.numel() == grad_outputs.numel(), "Grad tensor must be a CUDA tensor!");

    grad_outputs = grad_outputs.contiguous(); // Apparently since the loss output might be non-contiguous
    x = x.contiguous();
    auto grad_x = torch::empty_like(x).contiguous();
    int size = x.numel();

    // @TODO: Better kernel launch configuration
    int blockSize = 256;
    int num4 = size/4;
    int numBlocks = (num4 + blockSize - 1) / blockSize;

    sss_backward_kernel_vec<float><<<numBlocks, blockSize>>>(

        x.data_ptr<float>(),
        grad_outputs.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        size
    );

    int tail = size % 4; // remaining elements
    if (tail > 0) {
        int tailBlockSize = 32;  // small, enough for 1–3 elements
        int tailNumBlocks = (tail + tailBlockSize - 1) / tailBlockSize;

        sss_backward_tail_kernel<<<tailNumBlocks, tailBlockSize>>>(
            x.data_ptr<float>(), 
            grad_outputs.data_ptr<float>(),
            grad_x.data_ptr<float>(), 
            num4 * 4, 
            size
        );
    }

    return {grad_x};
}


// ===================================================================
// AUTOGRAD CLASS DEFINITIONS

torch::Tensor SSSAutograd::forward(AutogradContext *ctx, Tensor x) {
    ctx->save_for_backward({x});
    return forward_cuda(x);
}

variable_list SSSAutograd::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    return backward_cuda(x, grad_outputs[0]);
}

torch::Tensor SSSAutograd_f4::forward(AutogradContext *ctx, Tensor x) {
    ctx->save_for_backward({x});
    return forward_cuda_vec(x);
}

variable_list SSSAutograd_f4::backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    return backward_cuda_vec(x, grad_outputs[0]);
}