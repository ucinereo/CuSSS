#ifndef XSSS_IMPL_HPP
#define XSSS_IMPL_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class xSSSAutograd : public Function<xSSSAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x, Tensor a);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda(const Tensor &x, const Tensor &a);
std::vector<torch::Tensor> backward_cuda(const Tensor &x, const Tensor &a, const Tensor &grad_outputs);

#endif
