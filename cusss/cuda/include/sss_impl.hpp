#ifndef SSS_IMPL_HPP
#define SSS_IMPL_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class SSSAutograd : public Function<SSSAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda(Tensor &x);
std::vector<torch::Tensor> backward_cuda(Tensor &x, Tensor &grad_outputs);

class SSSAutograd_f4 : public Function<SSSAutograd_f4> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda_vec(Tensor &x);
std::vector<torch::Tensor> backward_cuda_vec(Tensor &x, Tensor &grad_outputs);
#ifdef __CUDACC__ // Such that only NVCC will read the following line
// Template kernel declaration
template <typename scalar_t>
__global__ void sss_forward_kernel(const scalar_t* x, scalar_t* output, int size);

// template <typename scalar_t>
// __global__ void sss_forward_kernel_conv(const scalar_t* x, scalar_t* output, int size);

template <typename scalar_t>
__global__ void sss_backward_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size);

template <typename scalar_t>
__global__ void sss_forward_kernel_vec(const scalar_t* x, scalar_t* output, int size);

template <typename scalar_t>
__global__ void sss_backward_kernel_vec(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int size);

template <typename scalar_t>
__global__ void sss_forward_tail_kernel(const scalar_t* x, scalar_t* output, int start, int size);

template <typename scalar_t>
__global__ void sss_backward_tail_kernel(const scalar_t* x, const scalar_t* grad_out, scalar_t* grad_x, int start, int size);

#endif
#endif
