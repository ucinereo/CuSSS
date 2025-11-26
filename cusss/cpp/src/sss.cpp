#include "sss.hpp"
#include "sss_impl.hpp"

#include <iostream>
#include <torch/script.h>

#include <ATen/ATen.h>
#include <torch/library.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// ----- Autograd-Enabled Ops -----

// C++ autograd Function wrapper
struct SSSFunction : public torch::autograd::Function<SSSFunction> {
    
    static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                              const at::Tensor& x) {
        ctx->save_for_backward({x});
        return sss_forward_cuda(x);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs) 
    {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto gy = grad_outputs[0];
        return {sss_backward_cuda(x, gy)};
    }
};

// This is what Autograd dispatch calls
at::Tensor sss_forward_autograd(const at::Tensor& x) {
    return SSSFunction::apply(x);
}

TORCH_LIBRARY(sss, m) {
    m.def("forward(Tensor x) -> Tensor");
}

// AUTOGRAD implementation (calls C++ autograd::Function)
TORCH_LIBRARY_IMPL(sss, Autograd, m) {
    m.impl("forward", sss_forward_autograd);
}


