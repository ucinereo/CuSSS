#pragma once
#include <ATen/ATen.h>

#include <torch/script.h>

using torch::autograd::tensor_list;

// Forward pass: computes
//   f(x) = x * (a * (x / (1 + |x|)) + 0.5)
at::Tensor xssslu_forward_cuda(const at::Tensor& x, const at::Tensor& a);

// Backward pass: computes
//   df/dx = (2 * (a * x + 1) * |x| + x^2 + 4 * a * x + 1) / (2 * (1 + |x|)^2)
//   df/da = x^2 / (1 + |x|)
// and returns grad_output * df/dx
// and accumulates local_grad_a into sum_grad_a
tensor_list xssslu_backward_cuda(const at::Tensor& x, const at::Tensor& a, const at::Tensor& grad_output);
