#pragma once
#include <ATen/ATen.h>

// Forward pass: computes
//   f(x) = 0.5 * (x / (1 + |x|)) + 0.5
at::Tensor sss_forward_cuda(const at::Tensor& x);

// Backward pass: computes
//   df/dx = 0.5 / (1 + |x|)^2
// and returns grad_output * df/dx
at::Tensor sss_backward_cuda(const at::Tensor& y, const at::Tensor& grad_output);
