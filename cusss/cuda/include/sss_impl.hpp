#ifndef SSS_IMPL_HPP
#define SSS_IMPL_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class SSSAutograd : public Function<SSSAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda(Tensor &x);
std::vector<torch::Tensor> backward_cuda(Tensor &x, Tensor &grad_outputs);

#endif
