#ifndef XSSS_AUTOGRAD_HPP
#define XSSS_AUTOGRAD_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class xSSS : public torch::CustomClassHolder {

public:
  xSSS() {}

  Tensor forward(Tensor x, Tensor a);

  std::vector<Tensor> __getstate__() { return {}; }

  void __setstate__(const std::vector<Tensor> &state) { return; }
};


#endif