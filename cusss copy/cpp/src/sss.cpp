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

// wrapper class which we expose to the API.
// torch::Tensor SSS::forward(Tensor x) {
//     return SSSAutograd::apply(x);
// }

// // wrapper for float4
// torch::Tensor SSS_f4::forward(Tensor x) {
//     return SSSAutograd_f4::apply(x);
// }

// TORCH_LIBRARY(sss, m) {
//     m.class_<SSS>("SSS")
//         .def(torch::init<>(), "", {})
//         .def("forward", &SSS::forward)
//         .def_pickle(
//             [](const c10::intrusive_ptr<SSS> &self)
//                 -> std::vector<torch::Tensor> { return self->__getstate__(); },
//             [](const std::vector<torch::Tensor> &state)
//                 -> c10::intrusive_ptr<SSS> {
//             auto obj = c10::make_intrusive<SSS>();
//             obj->__setstate__(state);
//             return obj;
//             });
//     m.def("forward_impl", &forward_cuda);
//     m.def("backward_impl", &backward_cuda);

//     // float4 class
//     m.class_<SSS_f4>("SSS_f4")
//         .def(torch::init<>(), "", {})
//         .def("forward", &SSS_f4::forward)
//         .def_pickle(
//             [](const c10::intrusive_ptr<SSS_f4> &self) 
//                 -> std::vector<torch::Tensor> { return self->__getstate__(); },
//             [](const std::vector<torch::Tensor> &state) 
//                 -> c10::intrusive_ptr<SSS_f4> {
//             auto obj = c10::make_intrusive<SSS_f4>();
//             obj->__setstate__(state);
//             return obj;
//             });
//     m.def("forward_impl_f4", &forward_cuda_f4);
//     m.def("backward_impl_f4", &backward_cuda_f4);
// }


// Forward declaration of CUDA kernels
at::Tensor sss_forward_cuda(const at::Tensor& x);
at::Tensor sss_backward_cuda(const at::Tensor& x, const at::Tensor& grad_output);

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


