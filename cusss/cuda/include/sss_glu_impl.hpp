#ifndef SSS_GLU_IMPL_HPP
#define SSS_GLU_IMPL_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class SSSGLUAutograd : public Function<SSSGLUAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x, Tensor y);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

torch::Tensor forward_cuda(Tensor &x, Tensor &y);
std::vector<torch::Tensor> backward_cuda(Tensor &x, Tensor &y, Tensor &grad_outputs);
#ifdef __CUDACC__ // Such that only NVCC will read the following line
__global__ void sss_glu_forward_kernel(const float* x, const float* y, float* output, int size);
#endif
#endif
