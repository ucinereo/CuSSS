#include "sss.hpp"
#include "sss_impl.hpp"

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// wrapper class which we expose to the API.
torch::Tensor SSS::forward(Tensor x) {
    return SSSAutograd::apply(x);
}

// wrapper for float4
torch::Tensor SSS_f4::forward(Tensor x) {
    return SSSAutograd_f4::apply(x);
}

TORCH_LIBRARY(sss, m) {
    m.class_<SSS>("SSS")
        .def(torch::init<>(), "", {})
        .def("forward", &SSS::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<SSS> &self)
                -> std::vector<torch::Tensor> { return self->__getstate__(); },
            [](const std::vector<torch::Tensor> &state)
                -> c10::intrusive_ptr<SSS> {
            auto obj = c10::make_intrusive<SSS>();
            obj->__setstate__(state);
            return obj;
            });
    m.def("forward_impl", &forward_cuda);
    m.def("backward_impl", &backward_cuda);

    // float4 class
    m.class_<SSS_f4>("SSS_f4")
        .def(torch::init<>(), "", {})
        .def("forward", &SSS_f4::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<SSS_f4> &self)
                -> std::vector<torch::Tensor> { return self->__getstate__(); },
            [](const std::vector<torch::Tensor> &state)
                -> c10::intrusive_ptr<SSS_f4> {
            auto obj = c10::make_intrusive<SSS_f4>();
            obj->__setstate__(state);
            return obj;
            });
    m.def("forward_impl_f4", &forward_cuda_f4);
    m.def("backward_impl_f4", &backward_cuda_f4);
}
