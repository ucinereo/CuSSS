#ifndef SSS_AUTOGRAD_HPP
#define SSS_AUTOGRAD_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class SSS : public torch::CustomClassHolder {

public:
  SSS() {}

  Tensor forward(Tensor x);

  std::vector<Tensor> __getstate__() { return {}; }

  void __setstate__(const std::vector<Tensor> &state) { return; }
};

#endif