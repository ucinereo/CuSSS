#pragma once
#include <ATen/ATen.h>

#include <torch/script.h>

using torch::autograd::tensor_list;

// Forward pass: computes
//   f(x,y) = ((x / (1.0f + |x|)) * 0.5f + 0.5f) * x * y;
at::Tensor sssglu_forward_cuda(const at::Tensor& x, const at::Tensor& y);

// Backward pass: computes
//   [ inv = 1.0f / (1.0f + |x|) ]
//   df/dx = 0.5f * y * (x * inv * inv + x * inv + 1.0f);
//   df/dy = (x * inv * 0.5f + 0.5f) * x;
// and returns grad_output * df/dx
// and returns grad_output * df/dy
tensor_list sssglu_backward_cuda(const at::Tensor& x, const at::Tensor& y, const at::Tensor& grad_output);
