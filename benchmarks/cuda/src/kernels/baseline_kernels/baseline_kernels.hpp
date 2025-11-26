__global__ void sigmoid_forward_kernel(const float* x, float* output, int size);

__global__ void identity_kernel(const float* x, float* output, int size);
