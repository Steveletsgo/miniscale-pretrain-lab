#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

__global__ void vector_add(const float* left, const float* right, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = left[index] + right[index];
    }
}

int main() {
    constexpr int size = 1 << 20;
    constexpr std::size_t bytes = size * sizeof(float);
    std::vector<float> left(size, 2.0F);
    std::vector<float> right(size, 3.0F);
    std::vector<float> output(size, 0.0F);
    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(&device_left, bytes));
    CUDA_CHECK(cudaMalloc(&device_right, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(device_left, device_right, device_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

    float maximum_error = 0.0F;
    for (float value : output) {
        maximum_error = std::fmax(maximum_error, std::fabs(value - 5.0F));
    }
    std::printf("elements=%d max_error=%.1f\n", size, maximum_error);

    CUDA_CHECK(cudaFree(device_left));
    CUDA_CHECK(cudaFree(device_right));
    CUDA_CHECK(cudaFree(device_output));
    return maximum_error == 0.0F ? EXIT_SUCCESS : EXIT_FAILURE;
}
