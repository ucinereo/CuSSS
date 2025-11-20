#include "xsss.hpp"
#include "xsss_impl.hpp"

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// wrapper class which we expose to the API.
torch::Tensor xSSS::forward(Tensor x, Tensor a) {
    return xSSSAutograd::apply(x, a)[0];
}


TORCH_LIBRARY(sss, m) {
    m.class_<xSSS>("xSSS")
        .def(torch::init<>(), "", {})
        .def("forward", &xSSS::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<xSSS> &self)
                -> std::vector<torch::Tensor> { return self->__getstate__(); },
            [](const std::vector<torch::Tensor> &state)
                -> c10::intrusive_ptr<xSSS> {
            auto obj = c10::make_intrusive<xSSS>();
            obj->__setstate__(state);
            return obj;
            });
    m.def("forward_impl", &forward_cuda);
    m.def("backward_impl", &backward_cuda);

}
