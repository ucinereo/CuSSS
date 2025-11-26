#ifndef SSS_AUTOGRAD_HPP
#define SSS_AUTOGRAD_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

at::Tensor sss_forward_autograd(const at::Tensor& x);

#endif 