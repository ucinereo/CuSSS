#ifndef TEMPLATE_UTILS_HPP
#define TEMPLATE_UTILS_HPP

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/script.h>
#include <iostream>

// ===================================================================
// Helper for applying element-wise operations to vector types
template <typename vec_t, typename native_t, int N> struct VectorApplyHelper;

// Specialization for 4-element vectors (float4, double4)
template <typename vec_t, typename native_t>
struct VectorApplyHelper<vec_t, native_t, 4> {
  template <typename Op> __device__ static vec_t apply(const vec_t &v) {
    return {Op::forward(v.x), Op::forward(v.y), Op::forward(v.z),
            Op::forward(v.w)};
  }

  template <typename Op>
  __device__ static vec_t apply_backward(const vec_t &v, const vec_t &grad) {
    return {Op::backward(v.x, grad.x), Op::backward(v.y, grad.y),
            Op::backward(v.z, grad.z), Op::backward(v.w, grad.w)};
  }

  // xsss variants with parameter a
  template <typename Op>
  __device__ static vec_t apply_xsss(const vec_t &v, native_t a) {
    return {Op::forward(v.x, a), Op::forward(v.y, a), Op::forward(v.z, a),
            Op::forward(v.w, a)};
  }

  template <typename Op>
  __device__ static vec_t apply_backward_x(const vec_t &v, native_t a, const vec_t &grad) {
    return {Op::backward_x(v.x, a, grad.x), Op::backward_x(v.y, a, grad.y),
            Op::backward_x(v.z, a, grad.z), Op::backward_x(v.w, a, grad.w)};
  }

  template <typename Op>
  __device__ static native_t apply_backward_a(const vec_t &v, const vec_t &grad) {
    return Op::backward_a(v.x, grad.x) + Op::backward_a(v.y, grad.y) +
           Op::backward_a(v.z, grad.z) + Op::backward_a(v.w, grad.w);
  }
};

// Specialization for 2-element vectors (__half2, __nv_bfloat162)
template <typename vec_t, typename native_t>
struct VectorApplyHelper<vec_t, native_t, 2> {
  template <typename Op> __device__ static vec_t apply(const vec_t &v) {
    return {Op::forward(v.x), Op::forward(v.y)};
  }

  template <typename Op>
  __device__ static vec_t apply_backward(const vec_t &v, const vec_t &grad) {
    return {Op::backward(v.x, grad.x), Op::backward(v.y, grad.y)};
  }

  // xsss variants with parameter a
  template <typename Op>
  __device__ static vec_t apply_xsss(const vec_t &v, native_t a) {
    return {Op::forward(v.x, a), Op::forward(v.y, a)};
  }

  template <typename Op>
  __device__ static vec_t apply_backward_x(const vec_t &v, native_t a, const vec_t &grad) {
    return {Op::backward_x(v.x, a, grad.x), Op::backward_x(v.y, a, grad.y)};
  }

  template <typename Op>
  __device__ static native_t apply_backward_a(const vec_t &v, const vec_t &grad) {
    return Op::backward_a(v.x, grad.x) + Op::backward_a(v.y, grad.y);
  }
};

// Specialization for scalar types
template <typename vec_t, typename native_t>
struct VectorApplyHelper<vec_t, native_t, 1> {
  template <typename Op> __device__ static vec_t apply(const vec_t &v) {
    return Op::forward(v);
  }

  template <typename Op>
  __device__ static vec_t apply_backward(const vec_t &v, const vec_t &grad) {
    return Op::backward(v, grad);
  }
};

// ===================================================================
// Helper to convert scalar_t to native_t
template <typename scalar_t, typename native_t>
__device__ __forceinline__ native_t to_native(scalar_t val) {
  return static_cast<native_t>(val);
}

template <>
__device__ __forceinline__ __half to_native<c10::Half, __half>(c10::Half val) {
  return static_cast<__half>(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 to_native<c10::BFloat16, __nv_bfloat16>(c10::BFloat16 val) {
  return static_cast<__nv_bfloat16>(val);
}

// ===================================================================
// VectorIO Traits structs for double, float, half, bfloat16
template <typename scalar_t> struct VectorIO;

// Base helper to avoid repetition in VectorIO specializations
template <typename ScalarT, typename NativeT, typename VecT,
          typename ReductionT, int PackedSize>
struct VectorIOBase {
  using scalar_t = ScalarT;
  using native_t = NativeT;
  using vec_t = VecT;
  using reduction_t = ReductionT;

  const static int packed_size = PackedSize;

  template <typename Op> __device__ static vec_t apply(const vec_t &v) {
    return VectorApplyHelper<vec_t, native_t, packed_size>::template apply<Op>(
        v);
  }

  template <typename Op>
  __device__ static vec_t apply_backward(const vec_t &v, const vec_t &grad) {
    return VectorApplyHelper<vec_t, native_t,
                             packed_size>::template apply_backward<Op>(v, grad);
  }

  // xsss-specific methods
  template <typename Op>
  __device__ static vec_t apply_xsss(const vec_t &v, native_t a) {
    return VectorApplyHelper<vec_t, native_t,
                             packed_size>::template apply_xsss<Op>(v, a);
  }

  template <typename Op>
  __device__ static vec_t apply_backward_x(const vec_t &v, native_t a, const vec_t &grad) {
    return VectorApplyHelper<vec_t, native_t,
                             packed_size>::template apply_backward_x<Op>(v, a, grad);
  }

  template <typename Op>
  __device__ static native_t apply_backward_a(const vec_t &v, const vec_t &grad) {
    return VectorApplyHelper<vec_t, native_t,
                             packed_size>::template apply_backward_a<Op>(v, grad);
  }
};

template <>
struct VectorIO<float> : VectorIOBase<float, float, float4, float, 4> {};

template <>
struct VectorIO<double> : VectorIOBase<double, double, double4, double, 4> {};

template <>
struct VectorIO<c10::Half>
    : VectorIOBase<c10::Half, __half, __half2, float, 2> {};

template <>
struct VectorIO<c10::BFloat16>
    : VectorIOBase<c10::BFloat16, __nv_bfloat16, __nv_bfloat162, float, 2> {};

// template <> struct VectorIO<c10::Float8_e5m2>
//   : VectorIOBase<c10::Float8_e5m2, __nv_fp8_e5m2, __nv_fp8_e5m2, float, 1>
//   {};

inline at::Tensor fp8_to_float(const at::Tensor &x) {
  if (x.scalar_type() == torch::kFloat8_e5m2 ||
      x.scalar_type() == torch::kFloat8_e4m3fn) {
    return x.to(torch::kFloat);
  } else {
    return x;
  }
}

inline at::Tensor float_to_fp8(const at::Tensor &x, torch::Dtype dtype) {
  if (dtype == torch::kFloat8_e5m2 || dtype == torch::kFloat8_e4m3fn) {
    return x.to(dtype);
  } else {
    return x;
  }
}

inline int get_vector_size(torch::Dtype dtype) {
  switch (dtype) {
  case torch::kFloat:
  case torch::kDouble:
    return 4;
  case torch::kHalf:
  case torch::kBFloat16:
    return 2;
  case torch::kFloat8_e5m2:
    return 1;
  default:
    TORCH_CHECK(false, "Unsupported dtype for vectorization");
    return 1;
  }
}

#endif // TEMPLATE_UTILS_HPP