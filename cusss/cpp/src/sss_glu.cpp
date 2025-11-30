#include "sss_glu.hpp"
#include "sss_glu_impl.hpp"

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// wrapper class which we expose to the API.
torch::Tensor SSSGLU::forward(Tensor x, Tensor y) {
    return SSSGLUAutograd::apply(x, y);
}

TORCH_LIBRARY(sss_glu, m) {
    m.class_<SSSGLU>("SSSGLU")
        .def(torch::init<>(), "", {})
        .def("forward", &SSSGLU::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<SSSGLU> &self)
                -> std::vector<torch::Tensor> { return self->__getstate__(); },
            [](const std::vector<torch::Tensor> &state)
                -> c10::intrusive_ptr<SSSGLU> {
            auto obj = c10::make_intrusive<SSSGLU>();
            obj->__setstate__(state);
            return obj;
            });
    m.def("forward_impl", &forward_cuda);
    m.def("backward_impl", &backward_cuda);
}
