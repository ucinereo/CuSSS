__global__ void sigmoid_forward_kernel(const float* x, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float e = x[idx];
    output[idx] = __frcp_rn(1.0f + __expf(-e));
  }
}

__global__ void identity_kernel(const float* x, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = x[idx];
  }
}