#ifndef SSS_IMPL_HPP
#define SSS_IMPL_HPP

#include <iostream>
#include <torch/script.h>

#include <ATen/ATen.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// Forward pass: computes
//   f(x) = 0.5 * (x / (1 + |x|)) + 0.5
at::Tensor sss_forward_cuda(const at::Tensor& x);

// Backward pass: computes
//   df/dx = 0.5 / (1 + |x|)^2
// and returns grad_output * df/dx
at::Tensor sss_backward_cuda(const at::Tensor& x,
                             const at::Tensor& grad_output);


class SSSAutograd_f4 : public Function<SSSAutograd_f4> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda_f4(Tensor &x);
std::vector<torch::Tensor> backward_cuda_f4(Tensor &x, Tensor &grad_outputs);
#ifdef __CUDACC__ // Such that only NVCC will read the following line
__global__ void sss_forward_kernel(const float* x, float* output, int size);
#endif
#endif
