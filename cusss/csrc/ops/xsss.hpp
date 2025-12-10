#pragma once
#include <ATen/ATen.h>

at::Tensor xsss_forward_autograd(const at::Tensor& x, const at::Tensor& a);
