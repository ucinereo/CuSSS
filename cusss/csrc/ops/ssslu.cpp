#include "ssslu.hpp"
#include "ssslu_cuda.hpp"

#include <torch/extension.h>
#include <torch/library.h>

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

using torch::Tensor;

// To register a backward formula, we need to construct a custom autograd function
struct SSSLUFunction : public Function<SSSLUFunction> {

    static at::Tensor forward(AutogradContext *ctx, const at::Tensor& x) {
        // It is important that the forward looks like this.
        // If not, it can lead to the operator being silently not correct.
        // Source: https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#bookmark=id.gcevr8cskv86

        // We first need to construct a guard to disable AD dispatching for inplace or view operations.
        // Intuitively, this prevents the autograd engine from trying to record operations that would cause infinite recursion.
        // This is necessary to avoid infinite recursion in certain cases.
        at::AutoDispatchBelowADInplaceOrView guard;

        // Now we fetch the correct implementation.
        // To allow for backend-specific implementations, we use the dispatcher to find the correct implementation.
        // This also enables the use of JIT compilation and other optimizations for torchscript.
        static auto op = torch::Dispatcher::singleton()
            .findSchemaOrThrow("ssslu::forward", "")
            .typed<decltype(ssslu_forward_cuda)>();
        
        // We may save tensors or other data for backwards.
        ctx->save_for_backward({x});
        
        // Finally, we call the implementation.
        return op.call(x);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        // Retrieve the saved tensors
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto gy = grad_outputs[0];

        // Again, we need to request the dispatcher for the correct implementation.
        static auto op = torch::Dispatcher::singleton()
            .findSchemaOrThrow("ssslu::backward", "")
            .typed<decltype(ssslu_backward_cuda)>();

        return {op.call(x, gy)};
    }
};

// This is what autograd will call.
at::Tensor ssslu_forward_autograd(const at::Tensor& x) {
    return SSSLUFunction::apply(x);
}

// ===================================================================
// PyTorch Library Registration

// First we define the operator schema (independent of backend)
// This is what users will call.
TORCH_LIBRARY(ssslu, m) {
    m.def("forward(Tensor x) -> Tensor");
    m.def("backward(Tensor x, Tensor grad_out) -> Tensor");
}

// CUDA implementation (found via dispatcher if input is on CUDA)
TORCH_LIBRARY_IMPL(ssslu, CUDA, m) {
    m.impl("forward", ssslu_forward_cuda);
    m.impl("backward", ssslu_backward_cuda);
}

// AUTOGRAD implementation (found via dispatcher if autograd is enabled)
TORCH_LIBRARY_IMPL(ssslu, Autograd, m) {
    m.impl("forward", ssslu_forward_autograd);
    // Note that we don't need to register backward here. The autograd engine will call SSSLUFunction::backward automatically.
}

// CPU implementation (not implemented)
/*
TORCH_LIBRARY_IMPL(ssslu, CPU, m) {
    m.impl("forward", ssslu_forward_cpu);
}
*/



