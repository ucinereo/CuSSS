#ifndef SSSGLU_AUTOGRAD_HPP
#define SSSGLU_AUTOGRAD_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class SSSGLU : public torch::CustomClassHolder {

public:
  SSSGLU() {}

  Tensor forward(Tensor x, Tensor y);

  std::vector<Tensor> __getstate__() { return {}; }

  void __setstate__(const std::vector<Tensor> &state) { return; }
};
