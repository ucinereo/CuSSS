#include <cuda.h>
#include <torch/script.h>
#include <cuda_bf16.h>
// ===================================================================
template <typename T> struct sss_elementwise_op;

template <> struct sss_elementwise_op<float> {
  __device__ static float forward(float x) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    return (x * inv) * 0.5f + 0.5f;
  }

  __device__ static float backward(float x, float grad_output) {
    float inv = __frcp_rn(1.0f + fabsf(x));
    float grad_input = grad_output * 0.5f * inv * inv;
    return grad_input;
  }
};

template <> struct sss_elementwise_op<c10::Half> {
  __device__ static c10::Half forward(c10::Half x) {
    c10::Half one_half(0.5f);
    c10::Half one(1.0f);
    c10::Half inv = hrcp(static_cast<__half>(one + abs(x)));
    return (x * inv) * one_half + one_half;
  }

    __device__ static c10::Half backward(c10::Half x,
                                         c10::Half grad_output) {
        c10::Half one_half(0.5f);
        c10::Half one(1.0f);
        c10::Half inv = hrcp(static_cast<__half>(one + abs(x)));
        c10::Half grad_input = grad_output * one_half * inv * inv;
        return grad_input;
    }
};

template <> struct sss_elementwise_op<double> {
  __device__ static double forward(double x) {
    double inv = __drcp_rn(1.0 + fabs(x));
    return (x * inv) * 0.5 + 0.5;
  }

  __device__ static double backward(double x, double grad_output) {
    double inv = __drcp_rn(1.0 + fabs(x));
    double grad_input = grad_output * 0.5 * inv * inv;
    return grad_input;
  }
};

// template <> struct sss_elementwise_op_intr<c10::BFloat16> {
//   __device__ static c10::BFloat16 forward(c10::BFloat16 x) {
//     c10::BFloat16 one_half(0.5f);
//     c10::BFloat16 one(1.0f);
//     c10::BFloat16 inv = hrcp(static_cast<__nv_bfloat16>(one + abs(x)));
//     return (x * inv) * one_half + one_half;
//   }

//     __device__ static c10::BFloat16 backward(c10::BFloat16 x,
//                                              c10::BFloat16 grad_output) {
//         c10::BFloat16 one_half(0.5f);
//         c10::BFloat16 one(1.0f);
//         c10::BFloat16 inv = hrcp(static_cast<__nv_bfloat16>(one + abs(x)));
//         c10::BFloat16 grad_input = grad_output * one_half * inv * inv;
//         return grad_input;
//     }
// };

// for comparison of conversion vs. direct bf16 computation
template <> struct sss_elementwise_op<c10::BFloat16> {
  __device__ static c10::BFloat16 forward(c10::BFloat16 x) {
    float x_f = static_cast<float>(x);
    float inv = __frcp_rn(1.0f + fabsf(x));
    float result = (x_f * inv) * 0.5f + 0.5f;
    return static_cast<c10::BFloat16>(result);
  }

    __device__ static c10::BFloat16 backward(c10::BFloat16 x,
                                             c10::BFloat16 grad_output) {
        float x_f = static_cast<float>(x);
        float grad_output_f = static_cast<float>(grad_output);
        float inv = __frcp_rn(1.0f + fabsf(x_f));
        float grad_input = grad_output_f * 0.5f * inv * inv;
        return static_cast<c10::BFloat16>(grad_input);
    }
};

// Specialization for native __nv_bfloat16 (used in vectorized operations)
template <> struct sss_elementwise_op<__nv_bfloat16> {
  __device__ static __nv_bfloat16 forward(__nv_bfloat16 x) {
    float x_f = __bfloat162float(x);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float result = (x_f * inv) * 0.5f + 0.5f;
    return __float2bfloat16(result);
  }

  __device__ static __nv_bfloat16 backward(__nv_bfloat16 x,
                                           __nv_bfloat16 grad_output) {
    float x_f = __bfloat162float(x);
    float grad_output_f = __bfloat162float(grad_output);
    float inv = __frcp_rn(1.0f + fabsf(x_f));
    float grad_input = grad_output_f * 0.5f * inv * inv;
    return __float2bfloat16(grad_input);
  }
};


// ===================================================================
// VectorIO Traits structs for double, float, half, bfloat16
template <typename scalar_t> struct VectorIO;

template <> struct VectorIO<float> {
  using scalar_t = float;
  using native_t = float;
  using vec_t = float4;
  using reduction_t = float;

  const static int packed_size = 4;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2, scalar_t &x3) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
    x3 = v.w;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2,
                               scalar_t x3) {
    return {x0, x1, x2, x3};
  }
};

template <> struct VectorIO<double> {
  using scalar_t = double;
  using native_t = double;
  using vec_t = double4;
  using reduction_t = double;

  const static int packed_size = 4;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2, scalar_t &x3) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
    x3 = v.w;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2,
                               scalar_t x3) {
    return {x0, x1, x2, x3};
  }
};

template <> struct VectorIO<c10::Half> {
  using scalar_t = c10::Half;
  using native_t = __half;
  using vec_t = __half2;
  using reduction_t = float;

  const static int packed_size = 2;

  __device__ static void unpack2(const vec_t &v, scalar_t &x0, scalar_t &x1) {
    const native_t *ptr = reinterpret_cast<const native_t *>(&v);
    x0 = static_cast<scalar_t>(ptr[0]);
    x1 = static_cast<scalar_t>(ptr[1]);
  }

  __device__ static vec_t pack2(scalar_t x0, scalar_t x1) {
    vec_t v;
    native_t *ptr = reinterpret_cast<native_t *>(&v);
    ptr[0] = static_cast<native_t>(x0);
    ptr[1] = static_cast<native_t>(x1);
    return v;
  }
};

template <> struct VectorIO<c10::BFloat16> {
  using scalar_t = c10::BFloat16;
  using native_t = __nv_bfloat16;
  using vec_t = __nv_bfloat162;
  using reduction_t = float;

  const static int packed_size = 2;

  __device__ static void unpack2(const vec_t &v, scalar_t &x0, scalar_t &x1) {
    const native_t *ptr = reinterpret_cast<const native_t *>(&v);
    x0 = static_cast<scalar_t>(ptr[0]);
    x1 = static_cast<scalar_t>(ptr[1]);
  }

  __device__ static vec_t pack2(scalar_t x0, scalar_t x1) {
    vec_t v;
    native_t *ptr = reinterpret_cast<native_t *>(&v);
    ptr[0] = static_cast<native_t>(x0);
    ptr[1] = static_cast<native_t>(x1);
    return v;
  }
};