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
at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output);
                  
#endif
