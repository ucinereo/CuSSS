#pragma once
#include <ATen/ATen.h>

at::Tensor xssslu_forward_autograd(const at::Tensor& x, const at::Tensor& a);
