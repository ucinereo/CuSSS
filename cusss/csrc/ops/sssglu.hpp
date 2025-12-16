#pragma once
#include <ATen/ATen.h>

at::Tensor sssglu_forward_autograd(const at::Tensor& x, const at::Tensor& y);
