#pragma once
#include <ATen/ATen.h>

// Forward pass: computes
//   f(x) = x * (0.5 * (x / (1 + |x|)) + 0.5)
at::Tensor ssslu_forward_cuda(const at::Tensor& x);

// Backward pass: computes
//   df/dx =  ((2 + x) * |x| + x^2 + 2 * x + 1) / (2 * (1 + |x|)^2)
// and returns grad_output * df/dx
at::Tensor ssslu_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output);
